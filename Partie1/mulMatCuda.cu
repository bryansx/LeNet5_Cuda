#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>


#define MAX_ERR 1e-4

__global__ void MatrixMul(float *M1, float *M2, float *Mout, int n, int p) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    // Handling arbitrary vector size
    for(int k=0;k<p;k++) {
        Mout[i*p + j] += M1[i*n + k] * M2[k*n + j];

    }
}


void MatrixPrint(float *M, int n, int p)
{
    for(int i=0; i<n; i++) {
      for(int j=0;j<p;j++) {
          printf("%f |", M[i*n + j] );
      }
      printf ( "\n");
    }
}

void MatrixInit(float *M, int n, int p)
{
    srand(time(0));
    int lower = -1;
    for(int i=0; i<n; i++) {
      for(int j=0;j<p;j++) {
         M[i*n + j] = 2*((rand() % 2) + lower) + 1;
      }
    }
}

int main(int argc, char *argv[]){

    float *M1;
    float *M2;
    float *Mout;
    float *d_M1; 
    float *d_M2; 
    float *d_Mout;
    int M;
    int N;

    N = atoi(argv[2]);
    M = atoi(argv[1]);

    // Allocate memory
    M1 = (float*)malloc(sizeof(float) * N * M);
    M2 = (float*)malloc(sizeof(float) * N * M);
    Mout = (float*)malloc(sizeof(float) * N * M);

    // Initialize Matrix
    MatrixInit(M1, N,M);
    MatrixInit(M2, N,M);

    
    cudaMalloc((void**)&d_M1, sizeof(float)*N*M);
    cudaMalloc((void**)&d_M2, sizeof(float)*N*M);
    cudaMalloc((void**)&d_Mout, sizeof(float)*N*M);

    cudaMemcpy(d_M1, M1, sizeof(float) * N * M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * N * M, cudaMemcpyHostToDevice);

    // Main function
    int block_size = N;
    int grid_size = M;

    time_t t1 = clock();
    MatrixMul<<<grid_size,block_size>>>(d_M1, d_M2, d_Mout, N, M);
    time_t t2 = clock();

    cudaMemcpy(Mout, d_Mout, sizeof(float)*N * M, cudaMemcpyDeviceToHost);

    // Display the 2 matrix: Uncomment if you want to display small matrix
    //printf("M1:\n");
    //MatrixPrint(M1, N,M);
    //printf("M2: \n");
    //MatrixPrint(M2, N,M);

    //printf("M1 * M2 \n");
    //MatrixPrint(Mout, N,M);

    double time_spent = (double) 1000*(t2 - t1) / (CLOCKS_PER_SEC);

    printf("Computation duration: %f ms\n", time_spent);

    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    free(M1);
    free(M2);
    free(Mout);
}
