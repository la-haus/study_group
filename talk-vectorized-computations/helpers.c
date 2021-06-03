
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void randomize_array(float *array, int n) {
    for( int i=0; i< n; ++i) {
        array[i] = ((float) rand())/ RAND_MAX;
    }
}

double mean(double arr[], int n) {
    double accum = 0;
    for(int i = 0; i<n; i++) {
        accum += arr[i];
     }
    return accum / n;
}

double std(double arr[], int n) {
    double accum = 0.;
    double accum2 = 0.;

    for(int i = 0; i<n; i++) {
        double x = arr[i];
        accum += x;
        accum2 += x*x;
    }
    double _mean = accum / n;
    return sqrt(  accum2 / n  - _mean * _mean );
}

#define timeit( n_runs, code ) {\
    int _n = (n_runs); \
    double _samples[_n]; \
    printf("Running c code...:\n"); \
    for (int _i = 0; _i < _n; ++_i) { \
       clock_t _start_time = clock(); \
       (code); \
       clock_t _end_time = clock(); \
       double _elapsed_time_ms = ((double) (_end_time - _start_time)) * 1000.0 / CLOCKS_PER_SEC; \
       _samples[_i] = _elapsed_time_ms; \
       printf( "sample #%3d: %.3f ms\n", _i + 1, _samples[_i] ); \
    } \
    double _mean = mean(_samples, _n); \
    double _std = std(_samples, _n); \
    printf("\n%.3f ms ± %.3f ms per loop (mean ± std. dev. of %d runs)", \
           _mean, _std, _n); \
    }
