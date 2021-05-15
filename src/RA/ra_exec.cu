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

int exec_inverse(curafft_plan *plan){
    /*
    Two different execution flows
    */


    ///curafft_partial_conv workflow for insufficient memory


    /// curafft_conv workflow for enough memory
    checkCudaErrors(cudaMemset(plan->fw,0,plan->num_w*plan->nf1*plan->nf2*sizeof(CUCPX)));// this is needed
    // 1. convlution
    curafft_conv(plan);

    // 2. cufft
    int direction = plan->iflag;
    // cautious, a batch of fft, bath size is num_w when memory is sufficent.
    CUFFT_EXEC(plan->fftplan, plan->fw, plan->fw, direction);

    // 3. dft on w (or 1 dimensional nufft type3)

    // 4. deconvolution (correction)

}