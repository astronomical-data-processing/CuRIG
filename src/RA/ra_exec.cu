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

    checkCudaErrors(cudaMemset(plan->fw,0,plan->num_w*plan->nf1*plan->nf2*sizeof(CUCPX)));// this is needed
    // 1. convlution
    curafft_conv(plan);

    // 2. cufft
    int direction = plan->iflag;
    // cautious, a batch of fft, bath size is num_w
    CUFFT_EXEC(plan->fftplan, plan->fw, plan->fw, direction);


    // 3. deconvolution


}