/*
INVERSE: type 1

FORWARD: type 2

*/
#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>

#include "curafft_plan.h"
#include "conv_invoker.h"
#include "deconv.h"
#include "precomp.h"
#include "ragridder_plan.h"
#include "ra_exec.h"
#include "cuft.h"

__global__ void gridder_rescaling_complex(CUCPX *x, PCS scale_ratio, int N)
{
    /*
    scaling the arr x,    x[i]*scale_ratio
    Parameters:
        x - arr
        scale_ratio
        N - length of arr x
    */
    int idx;
    for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        x[idx].x *= scale_ratio;
        x[idx].y *= scale_ratio;
    }
}

__global__ void div_n_lm(CUCPX *fk, PCS xpixelsize, PCS ypixelsize, int N1, int N2)
{
    int idx;
    PCS n_lm;
    int row, col;
    for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < N1 * N2; idx += gridDim.x * blockDim.x)
    {
        row = idx / N1;
        col = idx % N1;
        n_lm = sqrt(1.0 - pow((row - N2 / 2) * xpixelsize, 2) - pow((col - N1 / 2) * ypixelsize, 2));
        fk[idx].x /= n_lm;
        fk[idx].y /= n_lm;
    }
}

int curaew_scaling(curafft_plan *plan, ragridder_plan *gridder_plan)
{
    // ending work  * 1/n

    int N1 = gridder_plan->width;
    int N2 = gridder_plan->height;
    int N = N1 * N2;
    int blocksize = 256;
    int gridsize = (N - 1) / blocksize + 1;

    div_n_lm<<<gridsize, blocksize>>>(plan->fk, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y, N1, N2);
    checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}

int exec_vis2dirty(curafft_plan *plan, ragridder_plan *gridder_plan)
{
    /*
    Currently, just suitable for improved W stacking
    Two different execution flows
        Flow1: the data size is relatively small and memory is sufficent for whole conv
        Flow2: the data size is too large, the data is divided into parts 
    */
    int ier = 0;
    //printf("execute flow %d\n",plan->execute_flow);
    if (plan->execute_flow == 1)
    {
        /// curafft_conv workflow for enough memory
#ifdef DEBUG
        printf("plan info printing...\n");
        printf("nf (%d,%d,%d), upsampfac %lf\n", plan->nf1, plan->nf2, plan->nf3, plan->copts.upsampfac);
        printf("gridder_plan info printing...\n");
        printf("fov %lf, current channel %d, w_s_r %lf\n", gridder_plan->fov, gridder_plan->cur_channel, gridder_plan->w_s_r);
#endif
        // 1. convlution
#ifdef TIME
        cudaEvent_t start, stop;
        float milliseconds = 0;
        float totaltime = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
#endif
        ier = curafft_conv(plan);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totaltime += milliseconds;
        printf("[time  ] conv time:\t\t %.3g s\n", milliseconds / 1000);
#endif
#ifdef DEBUG
        printf("conv result printing (first w plane)...\n");
        CPX *fw = (CPX *)malloc(sizeof(CPX) * plan->nf1 * plan->nf2 * plan->nf3);
        cudaMemcpy(fw, plan->fw, sizeof(CUCPX) * plan->nf1 * plan->nf2 * plan->nf3, cudaMemcpyDeviceToHost);
        PCS temp = 0;
        for (int i = 0; i < plan->nf2; i++)
        {
            for (int j = 0; j < plan->nf1; j++)
            {
                temp += fw[i * plan->nf1 + j].real();
                printf("%.3g ", fw[i * plan->nf1 + j].real());
            }
            printf("\n");
        }
        printf("fft 000 %.3g\n", temp);
#endif
        // printf("n1 n2 n3 M %d, %d, %d, %d\n",plan->nf1,plan->nf2,plan->nf3,plan->M);
        // 2. cufft
#ifdef TIME
        cudaEventRecord(start);
#endif
        int direction = plan->iflag;
        // cautious, a batch of fft, bath size is num_w when memory is sufficent.
        CUFFT_EXEC(plan->fftplan, plan->fw, plan->fw, direction); // sychronized or not
        cudaDeviceSynchronize();
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totaltime += milliseconds;
        printf("[time  ] cufft time:\t\t %.3g s\n", milliseconds / 1000);
#endif
#ifdef DEBUG
        printf("fft result printing (first w plane)...\n");
        //CPX *fw = (CPX *)malloc(sizeof(CPX)*plan->nf1*plan->nf2*plan->nf3);
        cudaMemcpy(fw, plan->fw, sizeof(CUCPX) * plan->nf1 * plan->nf2 * plan->nf3, cudaMemcpyDeviceToHost);
        for (int i = 0; i < plan->nf2; i++)
        {
            for (int j = 0; j < plan->nf1; j++)
                printf("%.3g ", fw[i * plan->nf1 + j].real());
            printf("\n");
        }
        temp = 0;
        for (int i = 0; i < plan->nf3; i++)
        {
            temp += fw[i * plan->nf1 * plan->nf2].real();
        }
        printf("dft 00 %.3g\n", temp);
#endif
        // keep the N1*N2*num_w. ignore the outputs that are out of range

        // 3. dft on w (or 1 dimensional nufft type3)
#ifdef TIME
        cudaEventRecord(start);
#endif
        curadft_invoker(plan, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totaltime += milliseconds;
        printf("[time  ] dft w time:\t\t %.3g s\n", milliseconds / 1000);
#endif
#ifdef DEBUG
        printf("part of dft result printing:...\n");
        //CPX *fw = (CPX *)malloc(sizeof(CPX)*plan->nf1*plan->nf2*plan->nf3);
        cudaMemcpy(fw, plan->fw, sizeof(CUCPX) * plan->nf1 * plan->nf2 * plan->nf3, cudaMemcpyDeviceToHost);
        for (int i = 0; i < plan->nf2; i++)
        {
            for (int j = 0; j < plan->nf1; j++)
                printf("%.3g ", fw[i * plan->nf1 + j].real());
            printf("\n");
        }
#endif
        // 4. deconvolution (correction)
        // error detected, 1. w term deconv
        // 1. 2D deconv towards u and v
#ifdef TIME
        cudaEventRecord(start);
#endif
        plan->dim = 2;
        ier = curafft_deconv(plan);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totaltime += milliseconds;
        printf("[time  ] 2d deconv time:\t\t %.3g s\n", milliseconds / 1000);
#endif
#ifdef DEBUG
        printf("deconv result printing stage 1:...\n");
        CPX *fk = (CPX *)malloc(sizeof(CPX) * plan->ms * plan->mt);
        cudaMemcpy(fk, plan->fk, sizeof(CUCPX) * plan->ms * plan->mt, cudaMemcpyDeviceToHost);
        for (int i = 0; i < plan->mt; i++)
        {
            for (int j = 0; j < plan->ms; j++)
                printf("%.5lf ", fk[i * plan->ms + j].real());
            printf("\n");
        }
#endif
        // 2. w term deconv on fk
#ifdef TIME
        cudaEventRecord(start);
#endif
        ier = curadft_w_deconv(plan, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totaltime += milliseconds;
        printf("[time  ] w term deconv time:\t\t %.3g s\n", milliseconds / 1000);
#endif
#ifdef DEBUG
        printf("deconv result printing stage 2:...\n");
        //CPX *fk = (CPX *)malloc(sizeof(CPX)*plan->ms*plan->mt);
        cudaMemcpy(fk, plan->fk, sizeof(CUCPX) * plan->ms * plan->mt, cudaMemcpyDeviceToHost);
        for (int i = 0; i < plan->mt; i++)
        {
            for (int j = 0; j < plan->ms; j++)
                printf("%.5lf ", fk[i * plan->ms + j].real());
            printf("\n");
        }
#endif
        // 5. ending work - scaling
        // /n_lm, fourier related rescale
#ifdef TIME
        cudaEventRecord(start);
#endif
        curaew_scaling(plan, gridder_plan);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totaltime += milliseconds;
        printf("[time  ] end work time:\t\t %.3g s\n", milliseconds / 1000);
        printf("[time  ] Total time:\t\t %.3g s\n", totaltime/1000);
#endif    
    }
    else if (plan->execute_flow == 2)
    {
        /// curafft_partial_conv workflow for insufficient memory

        // offset array with size of
        for (int i = 0; i < gridder_plan->num_w; i += plan->batchsize)
        {
            //memory allocation of fw may cause error, if size is too large, decrease the batchsize.
            checkCudaErrors(cudaMemset(plan->fw, 0, plan->batchsize * plan->nf1 * plan->nf2 * sizeof(CUCPX)));
            // 1. convlution
            curafft_conv(plan);
        }
    }
    return ier;
}