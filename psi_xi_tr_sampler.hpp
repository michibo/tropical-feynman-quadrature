/* 

Copyright (c) 2020   Michael Borinsky

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

*/


#pragma once

#include <cassert>
#include <memory>

using namespace std;

#include "graph.hpp"
#include "random.hpp"

// cut edge, r, mm, L
using sector_vector = vector< tuple< int, int, bool, int > >;

// cut edge, Jr, r, mm, L
using cut_tuple = tuple< int, double, int, bool, int >;

template< class Generator >
cut_tuple get_random_cut( 
        const graph& g, 
        const edge_subgraph_type& subgraph, 
        double JJr, 
        const J_vector& subgraph_table,
        Generator& gen 
        )
{
    assert( g.is_edge_subgraph(subgraph) );

    double R = true_random::uniform(gen) * JJr;
    double T = .0;

    cut_tuple cut;

    visit_cuts( g, subgraph, subgraph_table, 
        [R,&T,&cut]( int cut_edge, double Jr, int r, bool mm, int L )
        {
            T += Jr / r;

            cut = make_tuple( cut_edge, Jr, r, mm, L );

            if( T > R )
                return false;

            return true;
        }
    );

    assert( T > R );

    return cut;
}

template< class Generator >
void get_random_sector( 
        sector_vector& sector, 
        const graph& g, 
        double JJr,
        const J_vector& subgraph_table, 
        Generator& gen 
        )
{
    assert( g._E > 0 );
    assert( sector.size() == g._E );

    int r, L, cut_edge;
    bool mm;

    edge_subgraph_type subgraph = g.complete_edge_subgraph();

    for( int j = 0; j < g._E; j++ )
    {
        tie( cut_edge, JJr, r, mm, L ) = get_random_cut( g, subgraph, JJr, subgraph_table, gen );
        assert( subgraph[cut_edge] );

        subgraph.reset( cut_edge );

        sector[j] = make_tuple( cut_edge, r, mm, L );
    }

    assert( subgraph.count() == 0 );
}

template <class Generator>
pair< double, double > get_random_psi_xi_tropical_sample( 
        vector< double >& X, 
        const graph& g, 
        bool MM,
        int L, 
        const sector_vector& sector, 
        Generator& gen 
        )
{
    assert( X.size() == g._E );
    assert( sector.size() == g._E );

    int oL = L;

    double G  = 0.;
    double F  = 1.;
    double XX = 1.;

    for( int j = 0;; j++ )
    {
        int e, r, L;
        bool mm;
        tie(e, r, mm, L) = sector[j];

        assert( !( mm && !MM ) );

        if( MM && !mm )
            G = XX;

        MM = mm;

        X[e] = XX;

        if( L < oL )
        {
            F *= XX;
            oL = L;
        }

        if( j >= g._E - 1 )
            break;

        assert( r > 0 );

        double R = true_random::uniform( gen );
        double Y = pow( 1. - R, 1. / r );

        XX *= Y;
    }

    return make_pair(F, G*F);
}

