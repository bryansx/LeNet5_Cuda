#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define n 10000000
#define p 10000000
#define MAX_ERR 1e-6

__global__ void MatrixInit(float *M, int n, int p)
{
    srand(time(0));
    int lower = -1, upper = 1;
    int i, j;
    for(int i=0; i<n; i++) {
      for(int j=0;j<p;j++) {
         M[i*n + j] = (rand() % (upper - lower + 1)) + lower;
      }
    }
}


__global__ void cudaMatrixAdd(float *out, float *a, float *b, int n, int p) {
    for(int i=0; i<n; i++) {
      for(int j=0;j<p;j++) {
        out[i*n + p] = a[i] + b[i];
        }
    }
}

int main(){
    float *A, *B, *out;
    float *d_a, *d_b, *d_out;

    // Allocate memory
    A   = (float*)malloc(sizeof(float) * n * p);
    B   = (float*)malloc(sizeof(float) * n * p);
    out = (float*)malloc(sizeof(float) * n * p);

    // Initialize array
    MatrixInit(float A, n, p);
    MatrixInit(float B, n, p);

    cudaMalloc((void**)&d_a, sizeof(float)* n * p);
    cudaMalloc((void**)&d_b, sizeof(float)* n * p);
    cudaMalloc((void**)&d_out, sizeof(float)* n * p);

    cudaMemcpy(d_a, a, sizeof(float) * n * p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * n * p, cudaMemcpyHostToDevice);

    // Main function
    cudaMatrixAdd<<<1,1>>>(d_out, d_a, d_b, n * p);

    cudaMemcpy(out, d_out, sizeof(float)* n * p, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < n * p; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(out);
}
