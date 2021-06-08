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
    plan->kv.u = d_u;
    plan->kv.v = d_v;
    plan->kv.w = d_w;
    plan->kv.vis = d_c;

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
            CHECK(cudaMalloc(&plan->cell_loc, sizeof(INT_M) * M)); //need some where to be free
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
        checkCudaErrors(cudaMalloc(&plan->fw, plan->nf1 * plan->nf2 * plan->nf3 * sizeof(CUCPX)));
        checkCudaErrors(cudaMalloc(&plan->fwkerhalf1, (plan->nf1 / 2 + 1) * sizeof(PCS)));
        checkCudaErrors(cudaMalloc(&plan->fwkerhalf2, (plan->nf2 / 2 + 1) * sizeof(PCS)));
        if (plan->w_term_method)
            checkCudaErrors(cudaMalloc(&plan->fwkerhalf3, (plan->nf3 / 2 + 1) * sizeof(PCS)));
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

__global__ void w_term_dft(CUCPX *fw, int nf1, int nf2, int N1, int N2, int batchsize)
{
    int idx;
    for (idx == threadIdx.x + blockIdx.x * blockDim.x; idx < N1 * N2; idx += gridDim.x * blockDim.x)
    {
        CUCPX omega;
        double z_t_2pi = 2 * PI * (z); //revise how to get z
        int i = 0;
        omega.x = cos(z_t_2pi * i);
        omega.y = sin(z_t_2pi * i);
        fw[idx] = fw[id] * omega;
        for (i = 1; i < batchsize; i++)
        {
            omega.x = cos(z_t_2pi * i);
            omega.y = sin(z_t_2pi * i);
            fw[idx] += fw[idx + i * nf1 * nf2] * omega;
        }
    }
}

void curadft_invoker(curafft_plan *plan)
{
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
        dim3 grid((N1 * N2 - 1) / num_thread + 1);
    w_term_dft<<<grid, block>>>(fw, nf1, nf2, N1, N2, batchsize);
    checkCudaErrors(cudaDeviceSynchronize());
    return;
}

int curafft_free(curafft_plan *plan)
{
    //free gpu memory like cell_loc
    int ier = 0;
    if (plan->opts.gpu_sort)
    {
        CHECK(cudaFree(plan->cell_loc));
    }
    return ier;
}