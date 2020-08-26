/* 

Copyright (c) 2020   Michael Borinsky

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

*/


#pragma once

#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>
#include <cassert>
#include <cstdint>
#include <climits>
#include <stdexcept>

class semi_dynamic_bitset
{
public:
    semi_dynamic_bitset( size_t size ) : _size( size )
    {
        assert( 0 <= size && size < sizeof(uint64_t)*CHAR_BIT );
        if( size > sizeof(uint64_t)*CHAR_BIT-1 )
            throw invalid_argument( "bitsets of size larger than 63 are not supported" );
    }

public:
    void set()
    {
        _data = ( 1UL << _size ) - 1;
    }

    void reset()
    {
        _data = 0;
    }

    void set( int n )
    {
        assert( 0 <= n && n < _size );

        _data |= 1ULL << n;
    }

    void reset( int n )
    {
        assert( 0 <= n && n < _size );

        _data &= ~(1ULL << n);
    }

private:
    uint64_t lexicographically_next_bit_permutation( uint64_t v )
    {
        // https://graphics.stanford.edu/~seander/bithacks.html

        uint64_t t = v | (v - 1); // t gets v's least significant 0 bits set to 1
        // Next set to 1 the most significant bit to change,
        // set to 0 the least significant ones, and add the necessary 1 bits.
        return (t + 1) | (((~t & -~t) - 1) >> ( ctz(v) + 1 ));
    }

public:
    bool next_permutation()
    {
        _data = lexicographically_next_bit_permutation( _data );

        if( ( _data & 1ULL << _size ) != 0 )
        {
            _data = 0;
            return false;
        }

        return true;
    }

public:
    bool operator[]( int n ) const
    {
        assert( 0 <= n && n < _size );

        return ( ( _data >> n ) & 1 ) != 0;
    }

    size_t size() const
    {
        return _size;
    }

private:
    static int pop_count( uint64_t v )
    {
        /*
         https://graphics.stanford.edu/~seander/bithacks.html

         A generalization of the best bit counting method to integers of bit-widths upto 128 (parameterized by type T) is this:

        v = v - ((v >> 1) & (T)~(T)0/3);                           // temp
        v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);      // temp
        v = (v + (v >> 4)) & (T)~(T)0/255*15;                      // temp
        c = (T)(v * ((T)~(T)0/255)) >> (sizeof(T) - 1) * CHAR_BIT; // count

        */
        v = v - ((v >> 1) & (uint64_t)~(uint64_t)0/3);                           // temp
        v = (v & (uint64_t)~(uint64_t)0/15*3) + ((v >> 2) & (uint64_t)~(uint64_t)0/15*3);      // temp
        v = (v + (v >> 4)) & (uint64_t)~(uint64_t)0/255*15;                      // temp

        return (uint64_t)(v * ((uint64_t)~(uint64_t)0/255)) >> (sizeof(uint64_t) - 1) * CHAR_BIT; // count
    }

    static int ctz( uint64_t v )
    {
        return pop_count( (v & -v) - 1 );
    }

public:
    int count() const
    {
        return pop_count( _data );
    }

    bool none() const
    {
        return _data == 0;
    }

public:
    uint64_t data() const
    {
        return _data;
    }

protected:
    size_t _size;
    uint64_t _data;
};

inline std::ostream& operator<<( std::ostream& os, const semi_dynamic_bitset& bitset )
{
    for( int j = 0; j < bitset.size(); j++ )
    {
        os << (bitset[j] ? "1" : "0");
    }

    return os;
}


using namespace std;

using edge_subgraph_type = semi_dynamic_bitset;
using vertex_subgraph_type = semi_dynamic_bitset;

class graph
{
public:
    using adjacency_vector = vector< vector< pair< int, int > > >;

    using edge_tuple = pair< pair<int, int>, int >;
    using edge_vector = vector< edge_tuple >;

public:
    graph( edge_vector edges ) : _edges(move(edges))
    {
        _E = _edges.size();

        for ( unsigned int j = 0; j < _E; j++ )
        {
            pair<int, int> edge;
            int k,l,c;
            tie(edge, c) = _edges[j];
            tie(k,l) = edge;

            if( c <= 0 )
                throw invalid_argument( "graph: edge weights must be larger than 0" );

            if( k < 0 || l < 0 )
                throw invalid_argument( "graph: vertex numbers must start from 0" );
            
            int Vp = max(k, l) + 1;
            if( Vp > _adjacencies.size() )
                _adjacencies.resize(Vp);

            _adjacencies[k].push_back( make_pair( j, l ) );
            _adjacencies[l].push_back( make_pair( j, k ) );
        }

        _V = _adjacencies.size();
        for( int i = 0; i < _V; i++ )
        {
            if( _adjacencies[i].empty() )
            {
                stringstream s;
                s << "graph: the graph has a singleton vertex " << i;
                throw invalid_argument( s.str() );
            }
        }
    }

public:
    vertex_subgraph_type complete_vertex_subgraph() const
    {
        semi_dynamic_bitset vertex_subgraph( _V );
        vertex_subgraph.set();
        
        return vertex_subgraph;
    }

    vertex_subgraph_type empty_vertex_subgraph() const
    {
        semi_dynamic_bitset vertex_subgraph( _V );
        vertex_subgraph.reset();
        
        return vertex_subgraph;
    }

    edge_subgraph_type complete_edge_subgraph() const
    {
        semi_dynamic_bitset edge_subgraph( _E );
        edge_subgraph.set();
        
        return edge_subgraph;
    }

    edge_subgraph_type empty_edge_subgraph() const
    {
        semi_dynamic_bitset edge_subgraph( _E );
        edge_subgraph.reset();
        
        return edge_subgraph;
    }

    bool is_vertex_subgraph( const vertex_subgraph_type& subgraph ) const
    {
        return subgraph.size() == _V;
    }

    bool is_edge_subgraph( const edge_subgraph_type& subgraph ) const
    {
        return subgraph.size() == _E;
    }

public:
    int _V, _E;
    edge_vector _edges;

    adjacency_vector _adjacencies;
};

inline ostream& operator<<( ostream& os, const graph& g )
{
    os << "G(";
    for( int j = 0; j < g._E; j++ )
    {
        pair<int, int> edge;
        int k,l,c;
        tie(edge, c) = g._edges[j];
        tie(k,l) = edge;
        os << "[{" << k << "," << l << "}," << c << "]";

        if( j != (g._E - 1) )
            os << ",";
    }
    os << ")";

    return os;
}

