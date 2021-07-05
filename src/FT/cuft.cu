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
    //int upsampfac = plan->copts.upsampfac;

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
        int n1 = plan->nf1;
        int n2 = 1;
        int n3 = 1;
        checkCudaErrors(cudaMalloc(&plan->fwkerhalf1, (plan->nf1 / 2 + 1) * sizeof(PCS)));
        if(plan->dim>1){
            checkCudaErrors(cudaMalloc(&plan->fwkerhalf2, (plan->nf2 / 2 + 1) * sizeof(PCS)));
            n2 = plan->nf2;
        }
        if(plan->dim>2){
            checkCudaErrors(cudaMalloc(&plan->fwkerhalf3, (plan->nf3 / 2 + 1) * sizeof(PCS)));
            n3 = plan->nf3;
        }
        checkCudaErrors(cudaMalloc(&plan->fw, n1 * n2 * n3 * sizeof(CUCPX)));
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

int curafft_free(curafft_plan *plan)
{
    //free gpu memory like cell_loc
    int ier = 0;
    if (plan->opts.gpu_sort)
    {
        checkCudaErrors(cudaFree(plan->cell_loc));
    }

    switch (plan->dim)
    {
    case 3:
        checkCudaErrors(cudaFree(plan->fwkerhalf3));
        checkCudaErrors(cudaFree(plan->d_w));
    case 2:
        checkCudaErrors(cudaFree(plan->fwkerhalf2));
        checkCudaErrors(cudaFree(plan->d_v));
    case 1:
        checkCudaErrors(cudaFree(plan->fwkerhalf1));
        checkCudaErrors(cudaFree(plan->d_u));
        checkCudaErrors(cudaFree(plan->d_c));
        checkCudaErrors(cudaFree(plan->fw));
        if(!plan->opts.gpu_conv_only)checkCudaErrors(cudaFree(plan->fk));

    default:
        break;
    }

    return ier;
}

//------------------------------Below this line, all are just for Radio astronomy-------------------------
__global__ void w_term_dft(CUCPX *fw, int nf1, int nf2, int nf3, int N1, int N2, PCS xpixelsize, PCS ypixelsize, int flag, int batchsize)
{
    /*
        Specified for radio astronomy
        W term dft output driven method
        the output of cufft is FFTW format// just do dft on the in range pixels
        //flag 
    */
    int idx;
    flag = 1.0;
    for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N1 * N2; idx += gridDim.x * blockDim.x)
    {
        int row = idx / N1;
        int col = idx % N1;
        int idx_fw = 0;
        int w1 = 0;
        int w2 = 0;

        w1 = col >= N1 / 2 ? col - N1 / 2 : nf1 + col - N1 / 2;
        w2 = row >= N2 / 2 ? row - N2 / 2 : nf2 + row - N2 / 2;
        idx_fw = w1 + w2 * nf1;
        CUCPX temp;
        temp.x = 0;
        temp.y = 0;
        // from N/2 to N/2, not 0 to N
        
        PCS z = sqrt(1 - pow(((row - N2/2) * xpixelsize),2) - pow(((col-N2/2) * ypixelsize),2)) - 1; // revise for >1
        // double z_t_2pi = 2 * PI * (z); w have been scaling to pirange
        for (int i = 0; i < batchsize; i++)
        {
            //for partial computing, the i should add a shift, and fw should change  
            temp.x += fw[idx_fw + i*nf1*nf2].x * cos(z * (i-nf3/2)/(PCS)nf3 * 2 * PI * flag) - fw[idx_fw + i*nf1*nf2].y * sin(z * (i-nf3/2)/(PCS)nf3 * 2 * PI *flag);
            temp.y += fw[idx_fw + i*nf1*nf2].x * sin(z * (i-nf3/2)/(PCS)nf3 * 2 * PI * flag) + fw[idx_fw + i*nf1*nf2].y * cos(z * (i-nf3/2)/(PCS)nf3 * 2 * PI *flag);
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
    int nf3 = plan->nf3;
    int N1 = plan->ms;
    int N2 = plan->mt;

    int batchsize = plan->batchsize;
    int flag = plan->mode_flag;
    int num_threads = 512;

    dim3 block(num_threads);
    dim3 grid((N1 * N2 - 1) / num_threads + 1);
    w_term_dft<<<grid, block>>>(plan->fw, nf1, nf2, nf3, N1, N2, xpixelsize, ypixelsize, flag, batchsize);
    checkCudaErrors(cudaDeviceSynchronize());
    return;
}
