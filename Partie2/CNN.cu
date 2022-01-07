#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void Matrix2DInitRand(float *M, int n, int p)
{
    srand(time(0));
    for(int i=0; i<n; i++) {
      for(int j=0;j<p;j++) {
         M[i*n + j] = (float)rand() / (float)RAND_MAX ;
      }
    }
}

void Matrix3DInitZero(float *M, int fm, int n, int p)
{
    for(int i=0; i<n; i++) {
      for(int j=0;j<p;j++) {
        for(int k=0;j<fm;k++) {
          M[i*p + j*fm + k] = 0;
        }
      }
    }
}

void Matrix3DInitRand(float *M, int fm, int n, int p)
{
  srand(time(0));
  for(int i=0; i<n; i++) {
    for(int j=0;j<p;j++) {
      for(int k=0;j<fm;k++) {
        M[i*p + j*fm + k] = (float)rand() / (float)RAND_MAX;
      }
    }
  }
}



void Matrix2DPrint(float *M, int n, int p)
{
    for(int i=0; i<n; i++) {
      for(int j=0;j<p;j++) {
          printf("%f |", M[i*n + j] );
      }
      printf ( "\n");
    }
}



int main(int argc, char *argv[]){

  // Initialisation de la matrice d'entrée
  float *raw_data;
  raw_data = (float*)malloc(sizeof(float) * 32 * 32);
  Matrix2DInitRand(raw_data, 32, 32);

  // Initialisation de la matrice de sortie de la première conv
  float *C1_data;
  C1_data = (float*)malloc(sizeof(float) * 6 * 28 * 28);
  Matrix3DInitZero(C1_data, 6, 28, 28);
   
  // Initialisation de la matrice de sortie du sous-echantillonnage
  float *S1_data;
  S1_data = (float*)malloc(sizeof(float) * 6 * 14 * 14);
  Matrix3DInitZero(S1_data, 6, 14, 14);
  
  // Initialisation de la matrice des premiers kernels
  float *C1_kernel;
  C1_kernel = (float*)malloc(sizeof(float) * 6 * 5 * 5);
  Matrix3DInitRand(C1_kernel, 6, 14, 14);

}