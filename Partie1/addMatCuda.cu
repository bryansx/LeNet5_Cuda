#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 10000
#define MAX_ERR 1e-4

__global__ void vector_add(float *d_out, float *d_a, float *d_b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Handling arbitrary vector size
    if (tid < n){
        d_out[tid] = d_a[tid] + d_b[tid];
    }
}

int main(int argc, char *argv[]){

    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    cudaMalloc((void**)&d_a, sizeof(float)*N);
    cudaMalloc((void**)&d_b, sizeof(float)*N);
    cudaMalloc((void**)&d_out, sizeof(float)*N);

    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Main function
    int block_size = atoi(argv[2]);
    int grid_size = atoi(argv[1]);

    int t1 = clock();
    vector_add<<<grid_size,block_size>>>(d_out, d_a, d_b, N);
    int t2 = clock();

    double time_spent = (double) 1000*(t2 - t1) / (CLOCKS_PER_SEC);

    printf("Computation duration: %f ms\n", time_spent);

    cudaMemcpy(out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("Grid size : %d\n", grid_size);
    printf("Block size : %d\n", block_size);
    //printf("out[0] = %f\n", out[0]);
    //printf("PASSED\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a);
    free(b);
    free(out);
}
