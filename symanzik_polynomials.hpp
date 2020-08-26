/* 

Copyright (c) 2020   Michael Borinsky

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

*/

#pragma once

#include <vector>
#include <Eigen/Dense>

#include "graph.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::LDLT;

double get_laplacian( 
        MatrixXd& L,
        const graph& g, 
        const vector<double>& X,
        const edge_subgraph_type& contracted_subgraph,
        int num_components,
        const vector<int>& contracted_subgraph_components_map
        )
{
    assert( X.size() == g._E );
    assert( contracted_subgraph_components_map.size() == g._V );

    int E = g._E;
    int V = num_components;
    
    assert( L.rows() == V-1 && L.cols() == V-1 );

    L.setZero( V-1, V-1 );

    double Lambda = 1.;
    for ( int j = 0; j < E; j++ )
    {
        if( contracted_subgraph[j] )
            continue;

        pair<int, int> edge;
        int k,l,c;
        tie(edge, c) = g._edges[j];
        tie(k,l) = edge;

        Lambda *= X[j];

        k = contracted_subgraph_components_map[k];
        l = contracted_subgraph_components_map[l];

        if( k == l )
            continue;

        if ( k < l )
            tie(k, l) = make_pair(l, k); 

        double x = 1. / X[j];

        L(l,l) += x;

        if( k == V - 1 ) // delete last row
            continue;

        L(k,k) +=  x;
        L(k,l) += -x;
        // only lower triangular part of the laplacian matters for ldlt. => Fill only lower triangular part.
    }

    return Lambda;
}

void get_P_matrix( 
        MatrixXd& P,
        const graph& g, 
        const vector< VectorXd >& momenta,
        int num_components,
        const vector<int>& contracted_subgraph_components_map
        )
{
    assert( contracted_subgraph_components_map.size() == g._V );
    if( momenta.size() != g._V )
    {
        stringstream s;
        s << "P-matrix calculation - the graph " << g << " has " << g._V << " vertices, but only " << momenta.size() << " momenta were provided. An incoming momentum for each vertex is needed. Give a zero incoming momentum for internal vertices.)";
        throw domain_error(s.str());
    }

    int V = num_components;
    P.setZero( V - 1, momenta[0].size() );
    
    for( int i = 0; i < g._V; i++ )
    {
        int v = contracted_subgraph_components_map[i];

        if( v == V-1 )
            continue;

        P.row(v) += momenta[i].transpose();
    }
}

double eval_psi_polynomial( 
        double Lambda, 
        const LDLT< MatrixXd >& ldlt 
        )
{
    double detL = ldlt.matrixL().determinant();
    double detD = ldlt.vectorD().prod();

    return detL * detL * detD * Lambda;
}

double eval_pphi_polynomial(
        const MatrixXd& P,
        const LDLT< MatrixXd >& ldlt
        )
{
    return ( P.transpose() * ldlt.solve( P ) ).trace();
}

double eval_M_polynomial( 
        const graph& g, 
        const vector<double>& masses_sqr,
        const vector<double>& X,
        const edge_subgraph_type& contracted_subgraph
        )
{
    if( masses_sqr.size() != g._E )
    {
        stringstream s;
        s << "m polynomial calculation - the graph " << g << " has " << g._E << " edges, but only " << masses_sqr.size() << " squared masses were given. A mass for each edge is needed. (It can be zero.)";
        throw domain_error(s.str());
    }
    
    assert( g._E == X.size() );

    double M = 0.;

    for( int j = 0; j < g._E; j++ )
    {
        if( !contracted_subgraph[j] )
            M += X[j] * masses_sqr[j];
    }

    return M;
}


