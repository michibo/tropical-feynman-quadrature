// Completed primitive phi^4 8-loop graphs from in arXiv:0801.2856
// P_{8,n} = periods8[n-1]

#include "graph.hpp"

constexpr int periods8[41][20][2] =  {
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 6}, {3, 5}, {3, 7}, {4, 6}, {4, 8}, {5, 7}, {5, 9}, {6, 8}, {6, 10}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 6}, {3, 5}, {3, 7}, {4, 6}, {4, 9}, {5, 7}, {5, 8}, {6, 8}, {6, 10}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 6}, {3, 5}, {3, 8}, {4, 6}, {4, 7}, {5, 7}, {5, 9}, {6, 8}, {6, 10}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 6}, {3, 5}, {3, 9}, {4, 6}, {4, 7}, {5, 7}, {5, 8}, {6, 8}, {6, 10}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 6}, {3, 5}, {3, 9}, {4, 6}, {4, 10}, {5, 7}, {5, 8}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 7}, {3, 5}, {3, 6}, {4, 6}, {4, 9}, {5, 7}, {5, 8}, {6, 8}, {6, 10}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 9}, {3, 5}, {3, 6}, {4, 6}, {4, 7}, {5, 7}, {5, 8}, {6, 8}, {6, 10}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 7}, {3, 5}, {3, 6}, {4, 6}, {4, 9}, {5, 8}, {5, 10}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 9}, {3, 5}, {3, 6}, {4, 6}, {4, 7}, {5, 8}, {5, 10}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 9}, {3, 5}, {3, 6}, {4, 6}, {4, 10}, {5, 7}, {5, 8}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 8}, {3, 5}, {3, 9}, {4, 6}, {4, 7}, {5, 6}, {5, 7}, {6, 8}, {6, 10}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 7}, {3, 5}, {3, 8}, {4, 6}, {4, 9}, {5, 6}, {5, 10}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 7}, {3, 5}, {3, 9}, {4, 6}, {4, 8}, {5, 6}, {5, 10}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 7}, {3, 5}, {3, 9}, {4, 6}, {4, 10}, {5, 6}, {5, 8}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 9}, {3, 5}, {3, 10}, {4, 6}, {4, 7}, {5, 6}, {5, 8}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 9}, {2, 3}, {2, 4}, {2, 10}, {3, 5}, {3, 6}, {4, 5}, {4, 6}, {5, 7}, {5, 8}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 6}, {2, 3}, {2, 4}, {2, 7}, {3, 5}, {3, 8}, {4, 5}, {4, 9}, {5, 6}, {5, 10}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 6}, {2, 3}, {2, 4}, {2, 7}, {3, 5}, {3, 9}, {4, 5}, {4, 10}, {5, 6}, {5, 8}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 6}, {2, 3}, {2, 4}, {2, 9}, {3, 5}, {3, 7}, {4, 5}, {4, 8}, {5, 6}, {5, 10}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 6}, {2, 3}, {2, 4}, {2, 9}, {3, 5}, {3, 7}, {4, 5}, {4, 10}, {5, 6}, {5, 8}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 7}, {2, 3}, {2, 4}, {2, 9}, {3, 5}, {3, 6}, {4, 5}, {4, 10}, {5, 6}, {5, 8}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 9}, {2, 3}, {2, 4}, {2, 10}, {3, 5}, {3, 6}, {4, 5}, {4, 7}, {5, 6}, {5, 8}, {6, 7}, {6, 8}, {7, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 6}, {3, 7}, {3, 8}, {4, 9}, {4, 10}, {5, 6}, {5, 7}, {5, 8}, {6, 7}, {6, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 6}, {3, 7}, {3, 8}, {4, 9}, {4, 10}, {5, 6}, {5, 7}, {5, 8}, {6, 9}, {6, 10}, {7, 9}, {7, 10}, {8, 9}, {8, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 6}, {3, 7}, {3, 10}, {4, 8}, {4, 9}, {5, 6}, {5, 7}, {5, 8}, {6, 7}, {6, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 6}, {3, 7}, {3, 9}, {4, 8}, {4, 10}, {5, 6}, {5, 7}, {5, 8}, {6, 9}, {6, 10}, {7, 9}, {7, 10}, {8, 9}, {8, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 6}, {3, 7}, {3, 9}, {4, 8}, {4, 10}, {5, 6}, {5, 7}, {5, 8}, {6, 9}, {6, 10}, {7, 8}, {7, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 6}, {3, 7}, {3, 9}, {4, 8}, {4, 10}, {5, 6}, {5, 7}, {5, 8}, {6, 9}, {6, 10}, {7, 8}, {7, 10}, {8, 9}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 9}, {3, 6}, {3, 8}, {4, 7}, {4, 10}, {5, 6}, {5, 7}, {5, 8}, {6, 7}, {6, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 10}, {3, 6}, {3, 8}, {4, 7}, {4, 9}, {5, 6}, {5, 7}, {5, 8}, {6, 7}, {6, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 9}, {3, 6}, {3, 7}, {4, 8}, {4, 10}, {5, 6}, {5, 7}, {5, 8}, {6, 9}, {6, 10}, {7, 9}, {7, 10}, {8, 9}, {8, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 6}, {2, 8}, {3, 7}, {3, 9}, {4, 5}, {4, 6}, {4, 8}, {5, 7}, {5, 9}, {6, 7}, {6, 10}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 6}, {2, 8}, {3, 7}, {3, 9}, {4, 5}, {4, 6}, {4, 8}, {5, 7}, {5, 10}, {6, 7}, {6, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 6}, {2, 7}, {3, 8}, {3, 9}, {4, 5}, {4, 6}, {4, 7}, {5, 8}, {5, 10}, {6, 9}, {6, 10}, {7, 9}, {7, 10}, {8, 9}, {8, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 6}, {2, 9}, {3, 7}, {3, 8}, {4, 5}, {4, 6}, {4, 8}, {5, 7}, {5, 9}, {6, 7}, {6, 10}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 6}, {2, 8}, {3, 7}, {3, 9}, {4, 5}, {4, 6}, {4, 7}, {5, 8}, {5, 10}, {6, 9}, {6, 10}, {7, 8}, {7, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 6}, {2, 8}, {3, 7}, {3, 9}, {4, 5}, {4, 6}, {4, 7}, {5, 8}, {5, 10}, {6, 9}, {6, 10}, {7, 8}, {7, 10}, {8, 9}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 8}, {2, 3}, {2, 5}, {2, 6}, {3, 7}, {3, 9}, {4, 5}, {4, 6}, {4, 7}, {5, 8}, {5, 10}, {6, 9}, {6, 10}, {7, 8}, {7, 10}, {8, 9}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 4}, {1, 10}, {2, 3}, {2, 5}, {2, 6}, {3, 7}, {3, 8}, {4, 5}, {4, 6}, {4, 8}, {5, 7}, {5, 9}, {6, 7}, {6, 9}, {7, 10}, {8, 9}, {8, 10}, {9, 10}},
   {{1, 2}, {1, 3}, {1, 7}, {1, 8}, {2, 4}, {2, 5}, {2, 6}, {3, 4}, {3, 5}, {3, 6}, {4, 7}, {4, 8}, {5, 9}, {5, 10}, {6, 9}, {6, 10}, {7, 9}, {7, 10}, {8, 9}, {8, 10}},
   {{1, 2}, {1, 3}, {1, 6}, {1, 7}, {2, 4}, {2, 5}, {2, 9}, {3, 4}, {3, 5}, {3, 10}, {4, 6}, {4, 8}, {5, 7}, {5, 8}, {6, 9}, {6, 10}, {7, 9}, {7, 10}, {8, 9}, {8, 10}}
    };

graph get_phi4_8loop_graph( int num_period )
{
    // num_period can be a number from 1 to 41
    // as in Oliver Schnetz' notation arXiv:0801.2856

    int E = 20;
    int V = 10;

    graph::edge_vector edges;
    for ( int j = 0; j < E; j++ )
    {
        int k, l;
        k = periods8[num_period - 1][j][0];
        l = periods8[num_period - 1][j][1];

        if ( k != V && l != V ) // open the completed period
            edges.push_back( make_pair( make_pair( k-1, l-1), 1 ) );
        // vertex numbers start from 0 here!
    }

    return graph( edges );
}

