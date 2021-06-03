

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void randomize_array(float *array, int n) {
    for( int i=0; i< n; ++i) {
        array[i] = ((float) rand())/ RAND_MAX;
    }
}

double dot_product( float* a, float* b, int n ) {

    double result = 0;
    for( int i = 0; i < n; ++i ) {
        result += a[i] * b[i];
    }
    
    return result;
}


int measure_dot_product_time( float* a, float* b, int n ) {
    clock_t start_time = clock();
    float result = dot_product(a, b, n);
    clock_t end_time = clock();

    double elapsed_time_ms = ((double) (end_time - start_time)) * 1000.0 / CLOCKS_PER_SEC;
    printf( "\nN = %d\nelapsed time = %.3f ms\n", n, elapsed_time_ms ); 

    printf("\ndot_product: %f\n", result );

    return 0;    
}

int main() {
    int N = 1000000;
    // int N = (int) 1e8;

    float* a = (float*) malloc(N * sizeof(float));
    float* b = (float*) malloc(N * sizeof(float));

    randomize_array( a, N );
    randomize_array( b, N );

    measure_dot_product_time( a, b, N );

    free( a );
    free( b );
}