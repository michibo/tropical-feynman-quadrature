/* 

Copyright (c) 2020   Michael Borinsky

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

*/

#include <iomanip> 
#include <iostream>
#include <chrono>

using namespace std;

#include "graph.hpp"
#include "subgraph_table.hpp"
#include "feynman_integral.hpp"

#include "periods_8loop.hpp"
#include "periods_9loop.hpp"

vector< Eigen::VectorXd > get_symmetric_momenta( int V )
{
    // returns a symmetric momentum configuration 
    // such that p_i * p_j = delta_ij V^2 - V

    vector< Eigen::VectorXd > momenta(V); 
    // one in-coming momentum for each vertex
    for( int i = 0; i < V; i++ )
    {
        momenta[i].setZero(V);
        for( int j = 0; j < V; j++ )
        {
            if( i == j ) 
                momenta[i][j] = V - 1.;
            else 
                momenta[i][j] =   - 1.;
        }
    }

    return momenta;
}

double integrate_graph( const graph& g, int D, const vector< Eigen::VectorXd >& momenta, const vector<double>& masses_sqr, uint64_t N )
{
    cout << "Integrating graph " << g << endl;

    double mm_eps = 1e-10; 
    // mm_eps must be almost zero
    // it is used to decide if a subgraph is 
    // mass-momentum spanning.

    J_vector subgraph_table;

    int W; // superficial degree of divergence w(G)
    double IGtr; // tropicalized Feynman integral I_G^tr

    // time tracking
    chrono::time_point<chrono::system_clock> start, end;
    chrono::duration<double> elapsed_seconds;


    // Start with Jr subgraph table: (i.e. preprocessing step)
    cout << "Started calculating Jr-table" << endl;

    start = std::chrono::system_clock::now();
    tie(subgraph_table, W, IGtr) = generate_subgraph_table( g, D, momenta, masses_sqr, mm_eps );
    end = std::chrono::system_clock::now();

    elapsed_seconds = end-start;

    cout << "Finished calculating Jr-table in " << elapsed_seconds.count() << " seconds " << endl;
    cout << "Using " << subgraph_table.size() * sizeof(J_vector::value_type) << " bytes of RAM " << endl;

    // Tropical results:
    cout << "Superficial degree of divergence: " << W << endl;
    cout << "I^tr = " << IGtr << endl;


    // Initialize multithreading
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);

    // Initialize random number generator
    true_random::xoshiro256 gen( 0 );

    cout << "Start integrating using " << max_threads << " threads and N = " << (double)N << " points" <<  endl;

    start = std::chrono::system_clock::now();
    stats res = feynman_integral_estimate( N, g, D, momenta, masses_sqr, subgraph_table, gen );
    end = std::chrono::system_clock::now();

    elapsed_seconds = end-start;

    // Some performance statistics
    cout << "Finished sampling " << (double)N << " points in " << elapsed_seconds.count() << " seconds " << endl;
    cout << "Average speed: " << N/elapsed_seconds.count() << " samples / second " << endl;
    cout << "Relative accuracy: " << res.acc()/res.avg() << endl;

    // Tropically accelerated Monte Carlo results:
    cout << "I = " << res.avg() << " +/- " << res.acc() << endl;

    return res.avg();
}

//////////////////////////////////////////////////////////////////////////

void simple_example( uint64_t N = 1000000ULL )
{
    cout << "Example: Wheel with 3 spokes in phi^4 with D=4" << endl;

    constexpr int D = 4;

    // wheel with 3 spokes graph 
    graph g( graph::edge_vector{ 
        graph::edge_tuple{ {0, 1}, 1 }, 
        graph::edge_tuple{ {0, 2}, 1 }, 
        graph::edge_tuple{ {0, 3}, 1 }, 
        graph::edge_tuple{ {1, 2}, 1 }, 
        graph::edge_tuple{ {2, 3}, 1 }, 
        graph::edge_tuple{ {3, 1}, 1 }, 
    } );

    // symmetric in-coming momenta:
    vector< Eigen::VectorXd > momenta = get_symmetric_momenta( g._V );

    // no masses:
    vector<double> masses_sqr( g._E );
    for( int j = 0; j < g._E; j++ )
        masses_sqr[j] = 0;

    integrate_graph( g, D, momenta, masses_sqr, N );
    cout << "Exact result is 6 * zeta(3) = 7.21234141896" << endl;
}


// Integration of a massive Feynman graph with superficial degree of divergence 1
void massive_example( uint64_t N = 1000000ULL )
{
    cout << "Example: Mercedes graph with massive edges with D=4" << endl;

    constexpr int D = 4;

    // Mercedes graph 
    graph g( graph::edge_vector{ 
        graph::edge_tuple{ {0, 1}, 1 }, 
        graph::edge_tuple{ {0, 2}, 1 }, 
        graph::edge_tuple{ {0, 3}, 1 }, 
        graph::edge_tuple{ {1, 2}, 1 }, 
        graph::edge_tuple{ {2, 3}, 1 }, 
        graph::edge_tuple{ {3, 4}, 1 }, 
        graph::edge_tuple{ {4, 1}, 1 }
    } );

    // symmetric in-coming momenta:
    vector< Eigen::VectorXd > momenta = get_symmetric_momenta( g._V );

    // The dimension of the momenta is arbitrary and 
    // independent of the value of D.

    // asymmetric masses:
    vector<double> masses_sqr( g._E );
    for( int j = 0; j < g._E; j++ )
        masses_sqr[j] = j;

    integrate_graph( g, D, momenta, masses_sqr, N );
}

void wizard_example( uint64_t N = 1000000ULL )
{
    cout << "Example: 8 loop graph in phi^4 with D=4 (Brown-Schnetz wizard see Fig.3)" << endl;

    constexpr int D = 4;

    // Wizard graph 
    graph g( graph::edge_vector{ 
        graph::edge_tuple{ {1, 2}, 1 }, 
        graph::edge_tuple{ {2, 3}, 1 }, 
        graph::edge_tuple{ {3, 4}, 1 }, 
        graph::edge_tuple{ {5, 6}, 1 }, 
        graph::edge_tuple{ {6, 7}, 1 }, 
        graph::edge_tuple{ {7, 8}, 1 }, 
        graph::edge_tuple{ {1, 5}, 1 }, 
        graph::edge_tuple{ {2, 6}, 1 }, 
        graph::edge_tuple{ {3, 7}, 1 }, 
        graph::edge_tuple{ {4, 8}, 1 }, 
        graph::edge_tuple{ {5, 3}, 1 }, 
        graph::edge_tuple{ {2, 8}, 1 }, 
        graph::edge_tuple{ {6, 0}, 1 }, 
        graph::edge_tuple{ {7, 0}, 1 },
        graph::edge_tuple{ {0, 1}, 1 },
        graph::edge_tuple{ {0, 4}, 1 },
    } );

    // symmetric in-coming momenta:
    vector< Eigen::VectorXd > momenta = get_symmetric_momenta( g._V );

    vector<double> masses_sqr( g._E );
    for( int j = 0; j < g._E; j++ )
        masses_sqr[j] = 0;

    integrate_graph( g, D, momenta, masses_sqr, N );
}

void zigzag_example( int num_loops, uint64_t N = 1000000ULL )
{
    cout << "Example: Zigzag graph in phi^4 with D=4 and " << num_loops << " loops " << endl;

    constexpr int D = 4;

    int V = num_loops + 1;
    int E = 2 * num_loops;

    graph::edge_vector edges( E );

    for( int i = 0; i < num_loops-1; i++ )
    {
        edges[2*i+1] = graph::edge_tuple{ {  i, i+2}, 1 };
        edges[2*i+2] = graph::edge_tuple{ {i+1, i+2}, 1 };
    }

    edges[0] = graph::edge_tuple{ {0, 1}, 1 };
    edges[E-1] = graph::edge_tuple{ {0, V-1}, 1 };

    graph g( move(edges) );

    // symmetric in-coming momenta:
    vector< Eigen::VectorXd > momenta = get_symmetric_momenta( g._V );

    vector<double> masses_sqr( g._E );
    for( int j = 0; j < g._E; j++ )
        masses_sqr[j] = 0;

    double estimate = integrate_graph( g, D, momenta, masses_sqr, N );

    double zeta[] =  { 0., 0., 1.644934066848226e+00, 1.202056903159594e+00, 1.082323233711138e+00, 
        1.036927755143370e+00, 1.017343061984449e+00, 1.008349277381922e+00, 1.004077356197944e+00, 
        1.002008392826082e+00, 1.000994575127818e+00, 1.000494188604119e+00, 1.000246086553308e+00, 
        1.000122713347579e+00, 1.000061248135059e+00, 1.000030588236307e+00, 1.000015282259409e+00, 
        1.000007637197638e+00, 1.000003817293265e+00, 1.000001908212717e+00, 1.000000953962034e+00, 
        1.000000476932987e+00, 1.000000238450503e+00, 1.000000119219926e+00, 1.000000059608189e+00, 
        1.000000029803504e+00, 1.000000014901555e+00, 1.000000007450712e+00, 1.000000003725334e+00, 
        1.000000001862660e+00, 1.000000000931327e+00, 1.000000000465663e+00, 1.000000000232831e+00, 
        1.000000000116416e+00, 1.000000000058208e+00, 1.000000000029104e+00, 1.000000000014552e+00, 
        1.000000000007276e+00, 1.000000000003638e+00, 1.000000000001819e+00, 1.000000000000909e+00, 
        1.000000000000455e+00, 1.000000000000227e+00, 1.000000000000114e+00, 1.000000000000057e+00, 
        1.000000000000028e+00, 1.000000000000014e+00, 1.000000000000007e+00, 1.000000000000004e+00, 
        1.000000000000002e+00 };

    // See arXiv:1208.1890
    int n = num_loops;

    uint64_t C = 1; // Catalan number C(n-1)
    for ( int i = 0; i < n-1; i++ ) {
        C = C * 2 * (2 * i + 1) / (i + 2);
    }

    if( n % 2 == 0 )
    {
        uint64_t num = 4 * C;
        double res = num * zeta[2*n-3];

        cout << "Exact result " << num << " * zeta(" << 2*n - 3 << ") = " << res << endl;
        cout << "Actual deviation " << fabs(estimate - res) / res << endl;
    }
    else
    {
        uint64_t num = C * ( ( 1 << ( 2*n - 4 ) ) - 1 );
        uint64_t den = 1 << ( 2*n - 6 );
        double res = num/(double)den * zeta[2*n-3];

        cout << "Exact result " << num << "/" << den << " * zeta(" << 2*n - 3 << ") = " << res << endl;
        cout << "Actual deviation " << fabs(estimate - res) / res << endl;
    }
}


// Integration of any primitive 8 loop graph in phi^4
void period8_example( int num_period, uint64_t N = 1000000ULL )
{
    cout << "Example: phi^4 8 loop period " << num_period << " with D=4 " << endl;

    // num_period can be a number from 1 to 41
    // as in Oliver Schnetz' notation arXiv:0801.2856

    int D = 4;

    graph g = get_phi4_8loop_graph( num_period );

    vector< Eigen::VectorXd > momenta = get_symmetric_momenta( g._V );

    // masses:
    vector<double> masses_sqr( g._E );
    for( int j = 0; j < g._E; j++ )
        masses_sqr[j] = 0;
    
    integrate_graph( g, D, momenta, masses_sqr, N );
}

// Integration of any primitive 9 loop graph in phi^4
void period9_example( int num_period, uint64_t N = 1000000ULL )
{
    cout << "Example: phi^4 9 loop period " << num_period << " with D=4 " << endl;

    // num_period can be a number from 1 to 190
    // as in Oliver Schnetz' notation

    int D = 4;

    graph g = get_phi4_9loop_graph( num_period );

    vector< Eigen::VectorXd > momenta = get_symmetric_momenta( g._V );

    // masses:
    vector<double> masses_sqr( g._E );
    for( int j = 0; j < g._E; j++ )
        masses_sqr[j] = 0;
    
    integrate_graph( g, D, momenta, masses_sqr, N );
}

void phi4_13_loop_example( uint64_t N = 10000000ULL )
{
    cout << "Example: 13 loop graph in phi^4 with D=4 " << endl;

    constexpr int D = 4;

    graph g( graph::edge_vector{ 
        graph::edge_tuple{{13,3},1},
        graph::edge_tuple{{9,5},1},
        graph::edge_tuple{{11,6},1},
        graph::edge_tuple{{10,12},1},
        graph::edge_tuple{{3,6},1},
        graph::edge_tuple{{13,0},1},
        graph::edge_tuple{{5,0},1},
        graph::edge_tuple{{0,2},1},
        graph::edge_tuple{{3,1},1},
        graph::edge_tuple{{4,10},1},
        graph::edge_tuple{{6,5},1},
        graph::edge_tuple{{9,12},1},
        graph::edge_tuple{{1,11},1},
        graph::edge_tuple{{13,10},1},
        graph::edge_tuple{{13,2},1},
        graph::edge_tuple{{4,12},1},
        graph::edge_tuple{{7,1},1},
        graph::edge_tuple{{7,2},1},
        graph::edge_tuple{{4,2},1},
        graph::edge_tuple{{10,8},1},
        graph::edge_tuple{{7,12},1},
        graph::edge_tuple{{8,7},1},
        graph::edge_tuple{{9,4},1},
        graph::edge_tuple{{11,9},1},
        graph::edge_tuple{{8,11},1},
        graph::edge_tuple{{1,0},1},
    } );

    // symmetric in-coming momenta:
    vector< Eigen::VectorXd > momenta = get_symmetric_momenta( g._V );

    // no masses:
    vector<double> masses_sqr( g._E );
    for( int j = 0; j < g._E; j++ )
        masses_sqr[j] = 0;

    integrate_graph( g, D, momenta, masses_sqr, N );
}

void phi4_17_loop_example( uint64_t N = 1000000000ULL )
{
    cout << "Example: 17 loop graph in phi^4 with D=4 " << endl;

    constexpr int D = 4;

    graph g( graph::edge_vector{ 
        graph::edge_tuple{{2,6},1},
        graph::edge_tuple{{7,14},1},
        graph::edge_tuple{{0,17},1},
        graph::edge_tuple{{14,1},1},
        graph::edge_tuple{{15,11},1},
        graph::edge_tuple{{12,1},1},
        graph::edge_tuple{{8,9},1},
        graph::edge_tuple{{5,6},1},
        graph::edge_tuple{{15,0},1},
        graph::edge_tuple{{12,8},1},
        graph::edge_tuple{{2,3},1},
        graph::edge_tuple{{16,3},1},
        graph::edge_tuple{{16,2},1},
        graph::edge_tuple{{14,17},1},
        graph::edge_tuple{{4,13},1},
        graph::edge_tuple{{1,9},1},
        graph::edge_tuple{{5,8},1},
        graph::edge_tuple{{13,5},1},
        graph::edge_tuple{{0,4},1},
        graph::edge_tuple{{10,9},1},
        graph::edge_tuple{{12,6},1},
        graph::edge_tuple{{8,17},1},
        graph::edge_tuple{{17,2},1},
        graph::edge_tuple{{4,10},1},
        graph::edge_tuple{{3,6},1},
        graph::edge_tuple{{0,7},1},
        graph::edge_tuple{{3,11},1},
        graph::edge_tuple{{14,11},1},
        graph::edge_tuple{{7,13},1},
        graph::edge_tuple{{15,7},1},
        graph::edge_tuple{{16,10},1},
        graph::edge_tuple{{4,16},1},
        graph::edge_tuple{{9,5},1},
        graph::edge_tuple{{1,15},1},
    } );

    // symmetric in-coming momenta:
    vector< Eigen::VectorXd > momenta = get_symmetric_momenta( g._V );

    // no masses:
    vector<double> masses_sqr( g._E );
    for( int j = 0; j < g._E; j++ )
        masses_sqr[j] = 0;

    integrate_graph( g, D, momenta, masses_sqr, N );
}

int main()
{
    cout << scientific;

    simple_example();

    cout << "################################" << endl << endl;

    cout << "Computing all zig zag periods from 3 to 12 loops" << endl;
    cout << "################################" << endl;
    for( int i = 3; i < 13; i++ )
    {
        zigzag_example( i );

        cout << "################################" << endl;
    }
    cout << "################################" << endl << endl;

    massive_example();

    cout << "################################" << endl << endl;

    wizard_example();

    cout << "################################" << endl << endl;

    period8_example( 41 );

    cout << "################################" << endl << endl;

    period9_example( 111 );

    cout << "################################" << endl << endl;

    phi4_13_loop_example();

    cout << "################################" << endl << endl;

    // only uncomment if you have more than 256 GB of memory available:

    //phi4_17_loop_example();

    cout << "################################" << endl;

    //zigzag_example( 17, 1000000000ULL );

    cout << "################################" << endl;

    return 0;
}

