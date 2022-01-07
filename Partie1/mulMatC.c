#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>


void MatrixInit(float *M, int n, int p)
{
    srand(time(0));
    int lower = -1, upper = 1;
    int i, j;
    for(int i=0; i<n; i++) {
      for(int j=0;j<p;j++) {
         M[i*n + j] = 2*((rand() % 2) + lower) + 1;
      }
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

void MatrixMul(float *M1, float *M2, float *Mout, int n, int p){
    for(int i=0; i<n; i++) {
      for(int j=0;j<p;j++) {
        Mout[i*p + j] = 0;
        for(int k=0;k<p;k++) {
          Mout[i*p + j] += M1[i*n + k] * M2[k*n + j];
        }
      }
    }
}

int main(int argc, char *argv[]){
    float *M1;
    float *M2;

    int N = atoi(argv[2]);
    int M = atoi(argv[1]);

    float *Mout;

    M1 = (float*)malloc(sizeof(float) * N * M);
    M2 = (float*)malloc(sizeof(float) * N * M);
    Mout = (float*)malloc(sizeof(float) * N * M);

    MatrixInit(M1, N,M);
    MatrixInit(M2, N,M);

    int t1 = clock();
    MatrixMul(M1, M2, Mout, N, M);
    int t2 = clock();

    // Display the 2 matrix: Uncomment if you want to display small matrix
    //printf("M1:\n");
    //MatrixPrint(M1, N,M);
    //printf("M2: \n");
    //MatrixPrint(M2, N,M);

    //printf("M1 * M2 \n");
    //MatrixPrint(Mout, N,M);

    double time_spent = (double) 1000*(t2 - t1) / (CLOCKS_PER_SEC);

    printf("Computation duration: %f ms\n", time_spent);

    free(M1);
    free(M2);
    free(Mout);

    return 0;
}
