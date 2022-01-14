#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

void charBckgrndPrint(char *str, float rgb){
  printf("\033[48;2;%d;%d;%dm", (int) rgb*255, (int) rgb*255, (int) rgb*255);
  printf("%s\033[0m",str);
}

void imgColorPrint(int height, int width, float *img){
  int row, col;
  char *str="  ";
  for(row=0; row<height; row++){
    for(col=0; col<width; col++){
      charBckgrndPrint(str,img[row*width + col]);
    }
    printf("\n");
  }
}

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
        for(int k=0;k<fm;k++) {
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
      for(int k=0;k<fm;k++) {
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

__global__ void cudaConv2D(float* M, float* kernel, float* Mout, int M_ligne, int M_colonne, int kernel_size, int nb_kernel, int Mout_ligne, int Mout_colonne){
    
    //Convolution d'une matrice par un kernel
    int lig = blockIdx.x;
    int col = threadIdx.x;

    float s = 0.0;

    if (lig < Mout_ligne && col < Mout_colonne)
    {
        int tot = M_ligne * M_colonne;

        for (int kernel_lig = 0; kernel_lig < kernel_size; kernel_lig++) {
            for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                for (int n_k = 0; n_k < nb_kernel; n_k++)
                {
                    s += M[(lig + kernel_lig) * M_colonne + col + kernel_col + n_k * tot] * kernel[kernel_lig * kernel_size + kernel_col + n_k * nb_kernel];
                }
            }
        }
        Mout[lig * Mout_colonne + col] = s;
    }
}

__global__ void cudaMeanPool(float* M, float* Mout, int M_ligne, int M_colonne, int M_prof, int meanpool_size, int Mout_ligne, int Mout_colonne){
    /*
      cudaMeanPool réalise un sous-échantillonanage par moyennage 2x2 pixels vers 1 pixel.
    */

    // On effectuera les calculs sur les lignes et colonnes paires. 
    // ATTENTION: Dans ce cas blockIdx.x et threadIdx.x vont de 0 à 13 donc lig et col de 0 à 26.
    int lig = 2*blockIdx.x;
    int col = 2*threadIdx.x;


        float s;
        int tot_meanpool = meanpool_size * meanpool_size;
        int tot_M = M_ligne * M_colonne;
        int tot_Mout = Mout_ligne * Mout_colonne;
        
        for (int n_prof = 0; n_prof < M_prof; n_prof++){
            s = 0.0;
            
            for (int meanpool_lig = 0; meanpool_lig < meanpool_size; meanpool_lig++) {
                for (int meanpool_col = 0; meanpool_col < meanpool_size; meanpool_col++) {
                    s += M[(lig + meanpool_lig) * M_colonne + col + meanpool_col + n_prof * tot_M] / tot_meanpool;
                }
            }

            Mout[blockIdx.x * Mout_colonne + threadIdx.x + n_prof * tot_Mout] = s;
            
        }
}

__global__ void activation_tanh(float* M, int M_ligne, int M_colonne, int M_prof, float* Mout){
    
    int lig = blockIdx.x;
    int col = threadIdx.x;

    int tot_M = M_ligne * M_colonne;
    
    for (int n_prof = 0; n_prof < M_prof; n_prof++){
        Mout[lig * M_colonne + col + n_prof * tot_M] = tanh(M[lig * M_colonne + col + n_prof * tot_M]);
    }

}




int main(int argc, char *argv[]){

  // Layer 1
  // Initialisation de la matrice d'entrée
  float *raw_data;
  raw_data = (float*)malloc(sizeof(float) * 32 * 32);
  Matrix2DInitRand(raw_data, 32, 32);

  //imgColorPrint(32, 32, raw_data);

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
  Matrix3DInitRand(C1_kernel, 6, 5, 5);

  // Copie des matrices dans la mémoire GPU afin d'effectuer les calculs du réseau
  
  float *d_raw_data, *d_C1_data, *d_C1_data_activated, *d_C1_kernel, *d_S1_data;

  cudaMalloc((void**)&d_raw_data, sizeof(float) * 32 * 32 * 1);
  cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 32 * 32 * 1, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_C1_kernel, sizeof(float) * 5 * 5 * 6);
  cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * 5 * 5 * 6, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_C1_data, sizeof(float) * 28 * 28 * 6);
  cudaMemcpy(d_C1_data, C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_C1_data_activated, sizeof(float) * 28 * 28 * 6);

  cudaMalloc((void**)&d_S1_data, sizeof(float) * 14 * 14 * 6);

  // Layer 2: Première conv2D (6 kernels de taille 5x5 => matrice de taille 6x28x28)
  cudaConv2D<<<28,28>>>(d_raw_data, d_C1_kernel, d_C1_data, 32, 32, 5, 6, 28, 28);
  cudaDeviceSynchronize();

  // Activation tanh
  activation_tanh<<<28,28>>>(d_C1_data, 28, 28, 6, d_C1_data_activated);
  cudaDeviceSynchronize();

  // Layer 3: Sous échantillionnage par moyennage 2x2
  cudaMeanPool<<<14,14>>>(d_C1_data_activated, d_S1_data, 28, 28, 6, 2, 14, 14);
  cudaDeviceSynchronize();

  // Copie du resultat GPU sur CPU
  cudaMemcpy(S1_data, d_S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyDeviceToHost);
  
  // Un petit printf pour vérifier que la matrice S1 est non nulle
  printf("S1_data: \n");
  Matrix2DPrint(S1_data, 14, 14*6);

  cudaFree(d_C1_data);
  cudaFree(d_S1_data);
  cudaFree(d_C1_kernel);
  cudaFree(d_raw_data);

  free(C1_data);
  free(S1_data);
  free(raw_data);
  free(C1_kernel);

}