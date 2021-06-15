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

int setup_plan(int nf1, int nf2, int nf3, int M, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, curafft_plan *plan)
{
    /* different dim will have different setting
    ----plan setting, and related memory allocation----
        nf1, nf2, nf3 - number of UPTS after upsampling
        M - number of NUPTS (num of vis)
        d_u, d_v, d_w - location
        d_c - value
  */
    int ier = 0;
    plan->d_u = d_u;
    plan->d_v = d_v;
    plan->d_w = d_w;
    plan->d_c = d_c;
    plan->mode_flag = 1; //CMCL mode
    int upsampfac = plan->copts.upsampfac;

    plan->nf1 = nf1;
    plan->nf2 = nf2;
    plan->nf3 = nf3;

    plan->M = M;

    //plan->maxbatchsize = 1;

    plan->byte_now = 0;
    // No extra memory is needed in nuptsdriven method (case 1)
    switch (plan->opts.gpu_gridder_method)
    {
    case 0:
    {
        if (plan->opts.gpu_sort)
        {
            checkCudaErrors(cudaMalloc(&plan->cell_loc, sizeof(INT_M) * M)); //need some where to be free
        }
    }
    case 1:
    {
        //shared memroy method
    }
    case 2:
    {
        //multi pass
    }
    break;

    default:
        std::cerr << "err: invalid method " << std::endl;
    }

    if (!plan->opts.gpu_conv_only)
    {
        int nf1 = plan->nf1;
        int nf2 = 1;
        int nf3 = 1;
        checkCudaErrors(cudaMalloc(&plan->fwkerhalf1, (plan->nf1 / 2 + 1) * sizeof(PCS)));
        if(plan->dim>1){
            checkCudaErrors(cudaMalloc(&plan->fwkerhalf2, (plan->nf2 / 2 + 1) * sizeof(PCS)));
            nf2 = plan->nf2;
        }
        if(plan->dim>2){
            checkCudaErrors(cudaMalloc(&plan->fwkerhalf3, (plan->nf3 / 2 + 1) * sizeof(PCS)));
            nf3 = plan->nf3;
        }
        checkCudaErrors(cudaMalloc(&plan->fw, nf1 * nf2 * nf3 * sizeof(CUCPX)));
        /* For multi GPU
    cudaStream_t* streams =(cudaStream_t*) malloc(plan->opts.gpu_nstreams*
      sizeof(cudaStream_t));
    for(int i=0; i<plan->opts.gpu_nstreams; i++)
      checkCudaErrors(cudaStreamCreate(&streams[i]));
    plan->streams = streams;
    */
    }

    return ier;
}

__global__ void w_term_dft(CUCPX *fw, int nf1, int nf2, int N1, int N2, PCS xpixelsize, PCS ypixelsize, int flag, int batchsize)
{
    int idx;
    for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N1 * N2; idx += gridDim.x * blockDim.x)
    {
        CUCPX omega;
        int row = idx / N1;
        int col = idx % N1;
        int idx_fw = 0;
        int w1 = 0;
        int w2 = 0;
        if (flag == 1)
        {
            w1 = row >= N1 / 2 ? row - N1 / 2 : nf1 + row - N1 / 2;
            w2 = col >= N2 / 2 ? col - N2 / 2 : nf2 + col - N2 / 2;
        }
        else
        {
            w1 = row >= N1 / 2 ? nf1 + row - N1 / 2 : row;
            w2 = col >= N2 / 2 ? nf2 + col - N2 / 2 : col;
        }
        idx_fw = w1 + w2 * nf1;
        CUCPX temp;
        temp.x = 0;
        temp.y = 0;
        PCS z = sqrt(1 - pow((row * xpixelsize),2) - pow((col * ypixelsize),2)) - 1; // revise for >1
        double z_t_2pi = 2 * PI * (z);
        for (int i = 0; i < batchsize; i++)
        {
            //for partial computing, the i should add a shift
            omega.x = fw[idx_fw].x * cos(z_t_2pi * i) - fw[idx_fw].y * sin(z_t_2pi * i);
            omega.y = fw[idx_fw].x * sin(z_t_2pi * i) + fw[idx_fw].y * cos(z_t_2pi * i);
            //
            temp.x += omega.x;
            temp.y += omega.y;
        }
        fw[idx_fw] = temp;
    }
}

void curadft_invoker(curafft_plan *plan, PCS xpixelsize, PCS ypixelsize)
{
    /*
        Specified for radio astronomy
        Input: 
            fw - the res after 2D-FT towards each w
        Output:
            fw - after dft (part/whole based on batchsize)
    */
    int nf1 = plan->nf1;
    int nf2 = plan->nf2;

    int N1 = plan->ms;
    int N2 = plan->mt;

    int batchsize = plan->batchsize;
    int flag = plan->mode_flag;
    int num_threads = 1024;

    dim3 block(num_threads);
    dim3 grid((N1 * N2 - 1) / num_threads + 1);
    w_term_dft<<<grid, block>>>(plan->fw, nf1, nf2, N1, N2, xpixelsize, ypixelsize, flag, batchsize);
    checkCudaErrors(cudaDeviceSynchronize());
    return;
}

int curafft_free(curafft_plan *plan)
{
    //free gpu memory like cell_loc
    int ier = 0;
    if (plan->opts.gpu_sort)
    {
        checkCudaErrors(cudaFree(plan->cell_loc));
    }
    checkCudaErrors(cudaFree(plan->fw));
    checkCudaErrors(cudaFree(plan->fk));
    checkCudaErrors(cudaFree(plan->d_u));
    checkCudaErrors(cudaFree(plan->d_v));
    checkCudaErrors(cudaFree(plan->d_w));
    checkCudaErrors(cudaFree(plan->d_c));

    switch (plan->dim)
    {
    case 3:
        checkCudaErrors(cudaFree(plan->fwkerhalf3));
    case 2:
        checkCudaErrors(cudaFree(plan->fwkerhalf2));
    case 1:
        checkCudaErrors(cudaFree(plan->fwkerhalf1));

    default:
        break;
    }

    return ier;
}