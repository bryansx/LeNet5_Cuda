#include <stdio.h>
#include <cuda.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>

#define MAX 100

__global__ void MatrixInit(int **M, int n, int p) {
    /* CUDA's random number library uses curandState_t to keep track of the seed value
        we will store a random state for every thread  */
    curandState_t state;

    for(int i=0; i<n; i++){
        for(int j=0; j<p; j++){
            

            /* we have to initialize the state */
            curand_init(time(NULL), /* the seed controls the sequence of random values that are produced */
                        0, /* the sequence number is only important with multiple cores */
                        0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                        &state);

            M[i][j] = *curand(&state) % 2;
        }
    }

}


int main(){
    /* allocate an int on the GPU */
    int **M[10][10];

    M = (float**)malloc(sizeof(float) * 100);

    /* invoke the GPU to initialize all of the random states */
    MatrixInit<<<1, 1>>>(M, 10, 10);

    /* copy the random number back */
    int x;
    cudaMemcpy(&x, M, 100*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Random number = %d.\n", M[10][10]);
    printf("Random number = %d.\n", M[5][4]);
    printf("Random number = %d.\n", M[9][6]);

    /* free the memory we allocated */
    cudaFree(M);

    cudaDeviceSynchronize();

    return 0;
}
