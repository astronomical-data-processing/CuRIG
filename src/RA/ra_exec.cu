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
#include "deconv_invoker.h"
#include "dft.h"

int exec_inverse(curafft_plan *plan){
    /*
    Currently, just for improved W stacking
    Two different execution flows
        Flow1: the data size is small and memory is sufficent for whole conv
        Flow2: the data size is large, the data is divided into parts 
    */

    if(plan->execute_flow==1){
        /// curafft_conv workflow for enough memory
        checkCudaErrors(cudaMemset(plan->fw,0,plan->num_w*plan->nf1*plan->nf2*sizeof(CUCPX)));
        // 1. convlution
        curafft_conv(plan);

        // 2. cufft
        int direction = plan->iflag;
        // cautious, a batch of fft, bath size is num_w when memory is sufficent.
        CUFFT_EXEC(plan->fftplan, plan->fw, plan->fw, direction); // sychronized or not
        // keep the N1*N2*num_w. ignore the outputs that are out of range 

        // 3. dft on w (or 1 dimensional nufft type3)
        curadft_invoker(plan);
        
        // 4. deconvolution (correction)
        curafft_deconv(plan);
        
    }
    else if(plan->execute_flow==2){
        /// curafft_partial_conv workflow for insufficient memory
        
        // offset array with size of 
        for(int i=0; i<plan->num_w; i+=plan->batchsize){
            //memory allocation of fw may cause error, if size is too large, decrease the batchsize.
            checkCudaErrors(cudaMemset(plan->fw,0,plan->batchsize*plan->nf1*plan->nf2*sizeof(CUCPX)));
            // 1. convlution
            curafft_conv(plan);

        }
    }
    

}