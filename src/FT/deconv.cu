/* 
invoke deconv related kernels
*/
#include "deconv.h"

__global__ int deconv_2d(int N1, int N2, CUCPX* plan->fw, CUCPX* plan->fk, PCS* plan->fwkerhalf1, PCS* plan->fwkerhalf2){
    int idx;
    int nmodes = N1*N2
    for(idx = blockIdx.x*blockDim.x + threadIdx.x; idx < nmodes; idx+=gridDim.x*blockDim.x){
        int k1 = i % N1;
		int k2 = i / N1;
		PCS kervalue = fwkerhalf1[abs(k1-N1/2)]*fwkerhalf2[abs(k2-N2/2)];
		fk[i].x = fw[i].x/kervalue;
		fk[i].y = fw[i].y/kervalue;
    }
}


int w_term_correction(){
    int ier = 0;
    // can not simply use fwkerhalf
    

    return ier;
}

int curafft_deconv(curafft_plan *plan){
    int ier = 0;
    int N1 = plan->ms;
    int N2 = plan->mt;
    // int batch_size = plan->batchsize;

    deconv_2d<<<(nmodes-1)/blocksize, blocksize>>>(N1, N2, plan->fw,plan->fk, plan->fwkerhalf1, plan->fwkerhalf2);
    // legendre_rule gpu

    ier = w_term_correction();

    return ier;
}