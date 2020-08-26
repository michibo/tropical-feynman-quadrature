/* 

Copyright (c) 2020   Michael Borinsky

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

*/


#include "graph.hpp"
#include "components.hpp"

#include "symanzik_polynomials.hpp"

// Jr, r, mm, L
using J_vector = vector< tuple< double, int, char, char > >;

template < class CutFunc >
void visit_cuts( 
        const graph& g, 
        const edge_subgraph_type& subgraph, 
        const J_vector& subgraph_table,
        CutFunc func 
    )
{
    assert( g.is_edge_subgraph(subgraph) );

    for( int j = 0; j < g._E; j++ )
    {
        if( !subgraph[j] )
            continue;

        edge_subgraph_type cutgraph = subgraph;
        cutgraph.reset(j);
       
        double Jr;
        bool mm;
        int r, L;
        tie(Jr, r, mm, L) = subgraph_table[cutgraph.data()];

        if( !func( j, Jr, r, mm, L ) )
            break;
    }
}

double subgraph_Jr_sum( 
        const graph& g, 
        const edge_subgraph_type& subgraph, 
        const J_vector& subgraph_table 
        )
{
    assert( g.is_edge_subgraph(subgraph) );

    double JJr = 0;
    visit_cuts( g, subgraph, subgraph_table, 
        [&JJr]( int cut_edge, double Jr, int r, bool mm, int L )
        {
            JJr += Jr / r;
            
            return true;
        }
    );

    return JJr;
}

int omega( const graph& g, int D, int L, const edge_subgraph_type& subgraph )
{
    assert( g.is_edge_subgraph(subgraph) );
    assert( D % 2 == 0 );

    int m = 0;
    for ( int j = 0; j < g._E; j++ )
    {
        if ( !subgraph[j] )
            continue;

        pair<int, int> edge;
        int c;
        tie(edge, c) = g._edges[j];

        m += c;
    }

    return m - D/2 * L;
}


tuple< J_vector, int, double > generate_subgraph_table( 
        const graph& g, 
        int D, 
        const vector< Eigen::VectorXd >& momenta, 
        const vector<double>& masses_sqr,
        double mm_eps
        )
{
    if( D % 2 != 0 )
    {
        stringstream s;
        s << "D = " << D << " is not implemented, yet. It would be easy to do so, though." << endl;
        s << "Sorry, only even dimensions are supported at the moment!";
        throw domain_error(s.str());
    }

    assert( g._E <= CHAR_BIT * sizeof(unsigned long long) );

    if( g._E > CHAR_BIT * sizeof(unsigned long long) - 1 )
    {
        stringstream s;
        s << "E = " << g._E << " this graph is too large for this implementation!";
        throw domain_error(s.str());
    }

    J_vector subgraph_table( 1ULL << g._E );

    vector<int> components_map( g._V );

    edge_subgraph_type cplt_subgraph = g.complete_edge_subgraph();

    int C = components( components_map, g, cplt_subgraph );
    int L = g._E - g._V + C;
    
    if( C != 1 )
    {
        stringstream s;
        s << "Error: the graph " << g << " is not connected.";
        throw domain_error(s.str());
    }

    int W = omega( g, D, L, cplt_subgraph );

    Eigen::MatrixXd P_buffer;
    Eigen::MatrixXd L_buffer;

    Eigen::LDLT< Eigen::MatrixXd > ldlt_buffer;

    vector<double> X_buffer( g._E );

    for( int j = 0; j < g._E; j++ )
        X_buffer[j] = 1;

    for( int n = 0; n <= g._E; n++ )
    {
        edge_subgraph_type subgraph = g.empty_edge_subgraph();

        for ( int j = 0; j < n; j++ )
            subgraph.set(j);

        do
        {
            assert( subgraph.count() == n );

            int cV = components( components_map, g, subgraph );

            get_P_matrix( P_buffer, g, momenta, cV, components_map );

            L_buffer.resize( cV - 1, cV - 1 );
            get_laplacian( L_buffer, g, X_buffer, subgraph, cV, components_map );
            ldlt_buffer.compute( L_buffer );

            double pphi = eval_pphi_polynomial( P_buffer, ldlt_buffer );
            double pmsqr = pphi + eval_M_polynomial( g, masses_sqr, X_buffer, subgraph );

            double Jr;
            int r;

            bool mm = fabs( pmsqr ) < mm_eps;
            int L = subgraph.count() - g._V + cV;

            if( n == 0 )
            {
                Jr = 1.0;
                r = 1;
            }
            else
            {
                Jr = subgraph_Jr_sum( g, subgraph, subgraph_table );

                r = omega( g, D, L, subgraph );

                if( mm )
                    r -= W;
            }

            if( r <= 0 && !( n == g._E ) )
            {
                stringstream s;
                s << "Error: the graph " << g << " has a subdivergence. The subgraph with edges ";
                for( int j = 0; j < g._E; j++ )
                    if(subgraph[j])
                        s << j << " ";

                s << " has an (effective) superficial degree of divergence of " << r;
                throw domain_error(s.str());
            }

            subgraph_table[subgraph.data()] = make_tuple( Jr, r, mm, L );
        }
        while( subgraph.next_permutation() );
    }

    double IGtr = get<0>(subgraph_table[cplt_subgraph.data()]);

    return make_tuple(move(subgraph_table), W, IGtr);
}

