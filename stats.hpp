/* 

Copyright (c) 2020   Michael Borinsky

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

*/


#include <array>
#include <cmath>

constexpr int N_SUM_LIMBS = 64;

class fp_summer
{
public:
    fp_summer()
    {
        for( double& s : _s )
            s = .0;
    }

    void add( double v )
    {
        if( v == .0 )
            return;

        bool bAdded = false;
        for( double& s : _s )
        {
            if( s == .0 )
            {
                s = v;

                bAdded = true;
                break;
            }

            v += s;
            s = .0;
        }

        if(!bAdded)
            throw runtime_error("ERROR: More limbs needed for fp_summer");
    }

    double value() const
    {
        double val = .0;

        for( double s : _s )
            val += s;

        return val;
    }

    array< double, N_SUM_LIMBS > _s;
};

fp_summer merge_summers( const fp_summer& entry1, const fp_summer& entry2 )
{
    assert( entry1._s[N_SUM_LIMBS-1] == .0 && entry2._s[N_SUM_LIMBS-1] == .0 );

    fp_summer m;

    for( int i = 0; i < N_SUM_LIMBS-1; i++ )
        m._s[i+1] = entry1._s[i] + entry2._s[i];

    m._s[0] = .0;

    return m;
}

class stats
{
public:
    stats()
    { }

    void update( double val )
    {
        N++;
        S1.add( val );
        S2.add( val*val );
    }

    double avg() const
    {
        return S1.value() / N;
    }

    double var() const
    {
        return ((double)N / ((double)N - 1)) * ( S2.value() / N - pow( avg(), 2 ) );
    }

    double std_def() const
    {
        return sqrt(var());
    }

    double acc() const
    {
        return sqrt(var()/N);
    }

public:
    uint64_t N = 0;

    fp_summer S1, S2;
};

stats merge_stats( const stats& mc1, const stats& mc2 )
{
    stats mc;
    mc.N = mc1.N + mc2.N;

    mc.S1 = merge_summers( mc1.S1, mc2.S1 );
    mc.S2 = merge_summers( mc1.S2, mc2.S2 );

    return mc;
}

stats merge_stats_vector( vector<stats> mcs )
{
    for( int e = 0; (1 << e) < mcs.size(); e++ )
    {
        for( int i = 0; i + (1 << e) < mcs.size(); i += 1 << (e+1) )
        {
            mcs[i] = merge_stats( mcs[i], mcs[i + (1 << e)] );
        }
    }

    return mcs[0];
}
