/* 

Copyright (c) 2020   Michael Borinsky

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

*/

#include "stats.hpp"

#include "random.hpp"
#include "psi_xi_tr_sampler.hpp"
#include "symanzik_polynomials.hpp"

template< class Generator >
stats feynman_integral_estimate( 
        uint64_t N, 
        const graph& g, 
        int D, 
        const vector< Eigen::VectorXd >& momenta, 
        const vector<double>& masses_sqr, 
        const J_vector& subgraph_table,
        Generator& gen
        )
{
    if( D % 2 != 0 )
    {
        stringstream s;
        s << "D = " << D << " is not implemented, yet. It would be easy to do so, though." << endl;
        s << "Sorry, only even dimensions are supported at the moment!";
        throw domain_error(s.str());
    }

    using namespace Eigen;

    edge_subgraph_type subgraph = g.complete_edge_subgraph();

    double IGtr;
    bool MM;
    int r, L;
    tie(IGtr, r, MM, L) = subgraph_table[subgraph.data()];
    assert(MM);

    edge_subgraph_type contracted_subgraph = g.empty_edge_subgraph();
    vector<int> contracted_subgraph_components_map( g._V );
    int C = components( contracted_subgraph_components_map, g, contracted_subgraph );

    int W = omega( g, D, L, subgraph );

    assert( C == g._V );

    MatrixXd P;
    get_P_matrix( P, g, momenta, C, contracted_subgraph_components_map );

    vector< true_random::xoshiro256 > generators; 

    int max_threads = omp_get_max_threads();

    for( int i = 0; i < max_threads; i++ )
    {
        generators.emplace_back( gen );
        gen.jump();
    }

    vector<stats> mcs( max_threads );

    MatrixXd         L_buffer( g._V-1, g._V-1 );
    LDLT< MatrixXd > ldlt_buffer( g._V-1 );

    sector_vector               sector_buffer( g._E );
    vector< double >                 X_buffer( g._E );

    #pragma omp parallel for default(none) \
        shared(cout,subgraph_table,mcs,generators,N,g,D,L,C,W,P,MM,masses_sqr,contracted_subgraph,contracted_subgraph_components_map,IGtr) \
        firstprivate(L_buffer,ldlt_buffer,sector_buffer,X_buffer) \
        schedule(dynamic, 10000)
    for( uint64_t i = 0; i < N; i++ )
    {
        int t = omp_get_thread_num();

        // generate a random sector labeled by a permutation using the table
        get_random_sector( sector_buffer, g, IGtr, subgraph_table, generators[t] );

        // generate the random point on the simplex and the values of psi_tr and xi_tr
        // using the permutation
        double psi_tr, xi_tr;
        tie(psi_tr, xi_tr) = get_random_psi_xi_tropical_sample( X_buffer, g, MM, L, sector_buffer, generators[t] );

        /// !

        // it is good for numerical stability to normalize the X variables 
        // such that psi_tr^(-D/2) * (psi_tr/xi_tr)^W = 1

        double tgt = pow(psi_tr, -D/2) * pow(psi_tr/xi_tr, W);
        double scale = pow( tgt, 1. / ( D/2 * L + W ) );

        for( int j = 0; j < g._E; j++ )
            X_buffer[j] *= scale;

        psi_tr = 1.0;
        xi_tr = 1.0;

        /// ! remove ! block if this is not needed

        double Lambda = get_laplacian( L_buffer, g, X_buffer, contracted_subgraph, C, contracted_subgraph_components_map );

        ldlt_buffer.compute( L_buffer );

        double psi = eval_psi_polynomial( Lambda, ldlt_buffer );
        
        double R = IGtr * pow( psi_tr/psi, D/2 );

        double pphi = eval_pphi_polynomial( P, ldlt_buffer );
        double xi  = psi * ( pphi + eval_M_polynomial( g, masses_sqr, X_buffer, contracted_subgraph ) );

        R *= pow( (psi / xi ) / (psi_tr / xi_tr), W );

        if( !isfinite( R ) )
        {
            #pragma omp critical
            {
                cout << "Warning: Sampled value " << R << " - floating point accuracy or numerical stabilty seem to be insufficient - dropping this point" << endl;
                cout << "If this happens often, the result will not be reliable!" << endl;

                for( int j = 0; j < g._E; j++ )
                    cout << "x_" << j << " = " << X_buffer[j] << "; ";
                cout << endl;

                cout << "Psi = " << psi << " ; Phi = " << pphi * psi << endl;
                cout << "-" << endl;
            }
        }
        else
            mcs[t].update( R );
    }

    return merge_stats_vector( mcs );
}

