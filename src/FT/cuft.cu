/*
1. nufft plan setting
2. 1D - dft for w term
*/
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <cuComplex.h>
#include "utils.h"
#include "cuft.h"

__global__ void w_term_dft(CUCPX *fw, int nf1, int nf2, int N1, int N2, int batchsize){
    int idx;
    for(idx == threadIdx.x + blockIdx.x * blockDim.x; idx < N1*N2; idx += gridDim.x*blockDim.x){
        CUCPX omega;
        double z_t_2pi = 2*PI*(z); //revise how to get z
        int i = 0;
        omega.x = cos(z_t_2pi*i);
        omega.y = sin(z_t_2pi*i);
        fw[idx] = fw[id]* omega;
        for(i=1; i<batchsize; i++){
            omega.x = cos(z_t_2pi*i);
            omega.y = sin(z_t_2pi*i);
            fw[idx] +=  fw[idx+i*nf1*nf2] * omega;
        }
    }
}

void curadft_invoker(curafft_plan *plan){
    /*
        Input: 
            fw - the res after 2D-FT towards each w
        Output:
            fw - after dft (part/whole based on batchsize) or save to fk
    */
    int nf1 = plan->nf1;
    int nf2 = plan->nf2;

    int N1 = plan->ms;
    int N2 = plan->mt;
    
    int batchsize = plan->batchsize;
    int num_threads = 1024;

    dim3 block(num_threads)
    dim3 grid((N1*N2-1)/num_thread+1);
    w_term_dft<<<grid,block>>>(fw,nf1,nf2,N1,N2,batchsize);
    checkCudaErrors(cudaDeviceSynchronize());
    return;
}