/* 

Copyright (c) 2020   Michael Borinsky

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

*/

#pragma once

#include <exception>

#include "graph.hpp"

#define MAX_BFS_STACKSIZE 32

#include <array>

template< class T, int MAX_STACK_SIZE >
class semi_dynamic_stack
{
public:
    semi_dynamic_stack() : _size(0)
    { }

    void push( T t )
    {
        assert( _size < MAX_STACK_SIZE );

        _data[_size++] = move(t);
    }

    T pop()
    {
        assert( _size > 0 );
        
        return move(_data[--_size]);
    }

    bool empty() const
    {
        return _size == 0;
    }

protected:
    size_t _size;
    array< T, MAX_STACK_SIZE > _data;
};


template < class VertexDiscoverFunc >
void bfs_discover_component( int v0, const graph& g, const edge_subgraph_type& subgraph, vertex_subgraph_type& discovered_vertices, VertexDiscoverFunc vertex_discover_func )
{
    assert( g._V <= MAX_BFS_STACKSIZE );

    assert( g.is_edge_subgraph(subgraph) );
    assert( g.is_vertex_subgraph(discovered_vertices) );

    semi_dynamic_stack< int, MAX_BFS_STACKSIZE > stack;

    auto discover_vertex = [&vertex_discover_func,&stack,&discovered_vertices] ( int v )
    {
        stack.push( v );
        discovered_vertices.set( v );

        vertex_discover_func( v );
    };

    discover_vertex( v0 );

    do
    {
        int v = stack.pop();
        assert( 0 <= v && v < g._V );

        for( const auto& adj : g._adjacencies[v] )
        {
            int j, l;
            tie(j, l) = adj;

            if( !subgraph[j] )
                continue;

            if( !discovered_vertices[l] )
                discover_vertex( l );
        }
    }
    while( !stack.empty() );
}

int components( vector<int>& components_map, const graph& g, const edge_subgraph_type& subgraph )
{
    assert( g.is_edge_subgraph( subgraph ) );
    assert( components_map.size() == g._V );

    if( g._V > MAX_BFS_STACKSIZE )
        throw std::runtime_error("Error: stacksize for BFS stack too small. Compile with larger MAX_BFS_STACKSIZE.");

    vertex_subgraph_type discovered_vertices = g.empty_vertex_subgraph();

    int C = 0;

    for( int i = 0; i < g._V; i++ )
    {
        if( discovered_vertices[i] )
            continue;

        bfs_discover_component( i, g, subgraph, discovered_vertices,
                [&components_map, C]( int v )
                {
                    components_map[v] = C;
                }
            );

        C++;
    }

    return C;
}

