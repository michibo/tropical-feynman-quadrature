/* 

Copyright (c) 2020   Michael Borinsky

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

*/

#include <iomanip> 
#include <iostream>

using namespace std;

#include "graph.hpp"
#include "subgraph_table.hpp"
#include "feynman_integral.hpp"

int main()
{
    cout << "Example: Wheel with 3 spokes in phi^4 with D=4" << endl;

    // We are going to integrate the 
    // wheel with 3 spokes graph 
    graph g( 
      graph::edge_vector{ 
        graph::edge_tuple{ {0, 1}, 1 }, 
        graph::edge_tuple{ {0, 2}, 1 }, 
        graph::edge_tuple{ {0, 3}, 1 }, 
        graph::edge_tuple{ {1, 2}, 1 }, 
        graph::edge_tuple{ {2, 3}, 1 }, 
        graph::edge_tuple{ {3, 1}, 1 }, 
      } 
    );

    // Notation for edges:
    //
    // the object graph::edge_tuple{ {v1, v2}, w }
    // represents an edge from vertex v1 to vertex v2
    // with edge weight w


    vector< Eigen::VectorXd > momenta{
        Eigen::Vector4d{   3.0, - 1.0, - 1.0, - 1.0 },
        Eigen::Vector4d{ - 1.0,   3.0, - 1.0, - 1.0 }, 
        Eigen::Vector4d{ - 1.0, - 1.0,   3.0, - 1.0 }, 
        Eigen::Vector4d{ - 1.0, - 1.0, - 1.0,   4.0 }
      };

    // one incoming momentum for each vertex of the graph 
    // is needed. An incoming momentum can be 0!
    
    // The momenta must be Euclidean! Minkowski vectors are 
    // not implemented (yet)
    

    vector<double> masses_sqr{ 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0 
      };

    // one squared mass for each edge of the graph is needed.


    // We consider a D = 4 integral:
    constexpr int D = 4;


    double mm_eps = 1e-10; 
    // mm_eps must be almost zero
    // it is used to decide if a subgraph is 
    // mass-momentum spanning.


    // This table stores the probabilities for each maximal 
    // cone of the braid arrangement fan:
    J_vector subgraph_table;


    int W; // superficial degree of divergence w(G)
    double IGtr; // tropicalized Feynman integral I_G^tr


    // Compute the Jr subgraph table: (i.e. perform the preprocessing step)
    tie(subgraph_table, W, IGtr) = generate_subgraph_table( g, D, momenta, masses_sqr, mm_eps );


    // 'Tropical' results:
    cout << "Superficial degree of divergence: " << W << endl;
    cout << "I^tr = " << IGtr << endl;


    // Initialize random number generator
    true_random::xoshiro256 gen( 0 );


    // Number of points to be sampled:
    constexpr uint64_t N = 1000000ULL;
  

    // Perform the actual Monte Carlo integration:
    stats res = feynman_integral_estimate( N, g, D, momenta, masses_sqr, subgraph_table, gen );


    // the res object stores the result:
    //
    // res.avg() gives the estimate,
    // res.acc() the estimated accuracy
    // res.var() the estimated sample variance

    // Monte Carlo results:
    cout << "I = " << res.avg() << " +/- " << res.acc() << endl;

    cout << "Relative accuracy: " << res.acc()/res.avg() << endl;

    return 0;
}

