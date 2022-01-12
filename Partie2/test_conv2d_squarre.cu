#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define HEIGHT 28
#define WIDTH 28

void charBckgrndPrint(char *str, float grayscale){
  printf("\033[48;2;%d;%d;%dm", (int) grayscale*255, (int) grayscale*255, (int) grayscale*255);
  printf("%s\033[0m",str);
}

void imgBinPrint(int height, int width, float *img){
  int row, col;
  char *str="  ";
  for(row=0; row<height; row++){
    for(col=0; col<width; col++){
        if(img[row*width + col] < 0){
            charBckgrndPrint(str,0);
        }
        else{
            charBckgrndPrint(str,img[row*width + col]);
        }
    }
    printf("\n");
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


int main(int argc, char *argv[]){
    // Nous allons faire un test de notre fonction de conv en appliquant
    // un filtre de SOBEL HORIZONTAL sur une image définie par nos soins

    int i, j;
    float *img;

    // Malloc image
    img = (float*)malloc(sizeof(float) * HEIGHT * WIDTH);

    for(i=0; i<HEIGHT; i++){
        for(j=0; j<WIDTH; j++){ 
            if(i==6 || i==20){
                img[i * HEIGHT + j] = 1;
            }
            if(j==6 || j==20){
                img[i * HEIGHT + j] = 1;
            }

        }
    }

    printf("Image test: \n");

    imgBinPrint(HEIGHT, WIDTH, img);

    // Creation du kernel de Sobel horizontal

    float *sobel_kernel;
    sobel_kernel = (float*)malloc(sizeof(float) * 3 * 3);
    sobel_kernel[0] = -1;
    sobel_kernel[1] = -2;
    sobel_kernel[2] = -1;
    sobel_kernel[3] = 0;
    sobel_kernel[4] = 0;
    sobel_kernel[5] = 0;
    sobel_kernel[6] = 1;
    sobel_kernel[7] = 2;
    sobel_kernel[8] = 1;

    //Creation des varibles dans GPU
    float *d_img, *d_sobel_kernel, *d_res;
    float *res;
    res = (float*)malloc(sizeof(float) * 26 * 26);

    cudaMalloc((void**)&d_res, sizeof(float) * 26 * 26 * 1);

    cudaMalloc((void**)&d_img, sizeof(float) * 28 * 28 * 1);
    cudaMemcpy(d_img, img, sizeof(float) * 28 * 28 * 1, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_sobel_kernel, sizeof(float) * 3 * 3 * 1);
    cudaMemcpy(d_sobel_kernel, sobel_kernel, sizeof(float) * 3 * 3 * 1, cudaMemcpyHostToDevice);


    // Do the convolution
    cudaConv2D<<<26,26>>>(d_img, d_sobel_kernel, d_res, 28, 28, 3, 1, 26, 26);
    cudaDeviceSynchronize();

    cudaMemcpy(res, d_res, sizeof(float) * 26 * 26 * 1, cudaMemcpyDeviceToHost);

    printf("Resultat de la conv: \n");
    imgBinPrint(26, 26, res);

    //Matrix2DPrint(res, 26, 26);

    cudaFree(d_res);
    cudaFree(d_img);
    cudaFree(d_sobel_kernel);

    free(res);
    free(img);
    free(sobel_kernel);



}