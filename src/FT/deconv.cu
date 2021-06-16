/* 
Deconvlution related kernels
    1D, 2D and 3D deconvlution
    Input is FFTW format (from 0 to N/2-1 and then -N/2 to -1), flag = 0
    Output is FFTW format or CMCL-compatible mode ordering (-N/2 to N/2-1), flag = 1
legendre_rule_fast cuda version should be implemented here
*/
#include "deconv.h"
#include "curafft_plan.h"

__global__ void deconv_1d(int N1, int nf1, CUCPX *fw, CUCPX *fk, PCS *fwkerhalf1, int flag){
    /*
        One dimensional deconvlution
        N - number of modes
        nf - grid size after upsampling
        fw - fft result
        fk - final result after deconv
        fwkerhalf - half of Fourier tranform (integal) of kernel fucntion, size - N/2+1
        flag - FFTW style (other) or CMCL (1)
    */
    int idx;
    int nmodes = N1;
    int w = 0;
    int k;
    int idx_fw = 0;
    for(idx = blockIdx.x*blockDim.x + threadIdx.x; idx < nmodes; idx+=gridDim.x*blockDim.x){
        k = idx;
        if(flag == 1){
            w = k >= N1/2 ? k - N1/2 : nf1 + k - N1/2; // CMCL
        }
        else{
            w = k >= N1/2 ? nf1+k-N1/2 : k; // FFTW
        }
        idx_fw = w;
        fk[idx].x = fw[idx_fw].x / fwkerhalf1[abs(k-N1/2)];
        fk[idx].y = fw[idx_fw].y / fwkerhalf1[abs(k-N1/2)];
    }
}

__global__ void deconv_2d(int N1, int N2, int nf1, int nf2, CUCPX* fw, CUCPX* fk, PCS* fwkerhalf1,
 PCS* fwkerhalf2, int flag){
     /*
        The output of cufft is from 0 to N/2-1 and then -N/2 to -1
        Should convert to -N/2 to N/2-1, then set flag = 1
     */
    int idx;
    int nmodes = N1*N2;
    int k1, k2, idx_fw, w1, w2;
    for(idx = blockIdx.x*blockDim.x + threadIdx.x; idx < nmodes; idx+=gridDim.x*blockDim.x){
        k1 = idx % N1;
		k2 = idx / N1;
        idx_fw = 0;
        w1 = 0;
        w2 = 0;
        if(flag == 1){
            w1 = k1 >= N1/2 ? k1-N1/2 : nf1+k1-N1/2;
		    w2 = k2 >= N2/2 ? k2-N2/2 : nf2+k2-N2/2;
        }
        else{
            w1 = k1 >= N1/2 ? nf1+k1-N1/2 : k1;
            w2 = k2 >= N2/2 ? nf2+k2-N2/2 : k2;
        }
        idx_fw = w1 + w2*nf1;
        
        
		PCS kervalue = fwkerhalf1[abs(k1-N1/2)]*fwkerhalf2[abs(k2-N2/2)];
		fk[idx].x = fw[idx_fw].x/kervalue;
		fk[idx].y = fw[idx_fw].y/kervalue;
    }
}

__global__ void deconv_3d(int N1, int N2, int N3, int nf1, int nf2, int nf3, CUCPX* fw, 
	CUCPX *fk, PCS *fwkerhalf1, PCS *fwkerhalf2, PCS *fwkerhalf3, int flag)
{
    int idx;
    int nmodes = N1*N2*N3;
    int k1, k2, k3, idx_fw, w1, w2, w3;
	for(idx=blockDim.x*blockIdx.x+threadIdx.x; idx<nmodes; idx+=blockDim.x*
		gridDim.x){
		k1 = idx % N1;
		k2 = (idx / N1) % N2;
		k3 = (idx / N1 / N2);
        w1=0, w2=0, w3=0;
        idx_fw = 0;
        if(flag == 1){
            w1 = k1 >= N1/2 ? k1-N1/2 : nf1+k1-N1/2;
		    w2 = k2 >= N2/2 ? k2-N2/2 : nf2+k2-N2/2;
		    w3 = k3 >= N3/2 ? k3-N3/2 : nf3+k3-N3/2;
        }
        else{
            w1 = k1 >= N1/2 ? nf1+k1-N1/2 : k1;
            w2 = k2 >= N2/2 ? nf2+k2-N2/2 : k2;
            w3 = k3 >= N3/2 ? nf3+k3-N3/2 : k3;
        }
	    idx_fw = w1 + w2*nf1 + w3*nf1*nf2;

		PCS kervalue = fwkerhalf1[abs(k1-N1/2)]*fwkerhalf2[abs(k2-N2/2)]*
			fwkerhalf3[abs(k3-N3/2)];
		fk[idx].x = fw[idx_fw].x/kervalue;
		fk[idx].y = fw[idx_fw].y/kervalue;
	}
}


int curafft_deconv(curafft_plan *plan){
    int ier = 0;
    int N1 = plan->ms;
    int nf1 = plan->nf1;
    int dim = plan->dim;
    int nmodes, N2, N3, nf2, nf3;
    // int batch_size = plan->batchsize;
    int flag = plan->mode_flag;
    int blocksize = 512;
    
    switch(dim){
        case 1:{
            nmodes = N1;
            deconv_1d<<<(nmodes-1)/blocksize, blocksize>>>(N1, nf1, plan->fw,plan->fk,
        plan->fwkerhalf1, flag);
            break;
        }
        case 2:{
            N2 = plan->mt;
            nf2 = plan->nf2;
            nmodes = N1*N2;
            deconv_2d<<<(nmodes-1)/blocksize, blocksize>>>(N1, N2, nf1, nf2, plan->fw,plan->fk,
        plan->fwkerhalf1, plan->fwkerhalf2, flag);
            break;
        }
        case 3:{
            N2 = plan->mt;
            N3 = plan->mu;
            nf2 = plan->nf2;
            nf3 = plan->nf3;
            nmodes = N1*N2*N3;
            deconv_3d<<<(nmodes-1)/blocksize, blocksize>>>(N1, N2, N3, nf1, nf2, nf3, plan->fw,plan->fk,
        plan->fwkerhalf1, plan->fwkerhalf2, plan->fwkerhalf3, flag);
            break;
        }
        default:{
            ier = 1; //error
        }
    }

    return ier;
}