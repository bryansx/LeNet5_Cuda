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

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for(int i=0; i<n; i++) {
      for(int j=0;j<p;j++) {
         Mout[i*n + j] = M1[i*n + j] + M2[i*n + j];
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
    
    //printf("M1:\n");
    //MatrixPrint(M1, N,M);
    //printf("M2: \n");
    //MatrixPrint(M2, N,M);

    //printf("M1 + M2 \n");

    int t1 = clock();
    MatrixAdd(M1, M2, Mout, N, M);
    int t2 = clock();

    printf("Delta T: %f \n", difftime(t2, t1));

    //MatrixPrint(Mout, N,M);

    free(M1);
    free(M2);
    free(Mout);

    return 0;
}
