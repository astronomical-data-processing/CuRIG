/* --------cugridder-----------
    1. gridder_setting
        fov and other astro related setting
        opt setting
        plan setting
        bin setting
    2. gridder_execution
    3. gridder_destroy
*/

#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>

#include "conv_invoker.h"
#include "ra_exec.h"
#include "utils.h"


int gridder_plan(){
    
}

// the bin sort should be completed at gridder_settting

int gridder_setting(int N1, int N2, int method, int kerevalmeth, int w_term_method, int direction, double sigma, int iflag,
    int batchsize, int M, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, curafft_plan *plan)
{
    /*
        N1, N2 - number of Fouier modes
        method - gridding method
        kerevalmeth - gridding kernel evaluation method
        direction - 1 CUFFT_INVERSE, 0 CUFFT_FORWARD
        sigma - upsampling factor
        iflag - flag for fourier transform
        batchsize - number of batch in  cufft (used for handling piece by piece)
        M - number of nputs (visibility)
        d_u, d_v, d_w - wavelengths in different dimensions
        d_c - value of visibility

        ****issue, degrid
    */
    int ier = 0;
    
    plan = new curafft_plan();
    memset(plan, 0, sizeof(curafft_plan));

    // fov and other astro related setting +++


    // opts and copts setting
    plan->opts.gpu_device_id = 0;
    plan->opts.upsampfac = sigma;
    plan->opts.gpu_sort = 1;
    plan->opts.gpu_binsizex = -1;
    plan->opts.gpu_binsizey = -1;
    plan->opts.gpu_binsizez = -1;
    plan->opts.gpu_kerevalmeth = kerevalmeth;
    plan->opts.gpu_conv_only = 0;
    plan->opts.gpu_gridder_method = method;

    int ier = setup_conv_opts(plan->copts, tol, sigma, kerevalmeth); //check the arguements

	if(ier!=0)printf("setup_error\n");

    // plan setting
    plan->w_term_method = w_term_method;
    // cuda stream malloc in setup_plan
    setup_plan(N1, N2, M, d_u, d_v, d_w, d_c, plan);
    // plan->dim = dim;
	plan->ms = N1;
	plan->mt = N2;
	// plan->mu = nmodes[2];

    int fftsign = (iflag>=0) ? 1 : -1;

	plan->iflag = fftsign;
    if (batchsize == 0) batchsize = min(4,plan->num_w);
	plan->batchsize = batchsize;

    plan->copts.direction = direction; // 1 inverse, 0 forward

    fwkerhalf1 = (PCS*)malloc(sizeof(PCS)*(nf1/2+1));
    onedim_fseries_kernel(nf1, fwkerhalf1, plan->spopts); // used for correction
    
    fwkerhalf2 = (PCS*)malloc(sizeof(PCS)*(nf2/2+1));
    onedim_fseries_kernel(nf2, fwkerhalf2, plan->spopts);
    
    if(w_term_method){
        // improved_ws
        fwkerhalf3 = (PCS*)malloc(sizeof(PCS)*(nf3/2+1));
        //need to revise
        onedim_fseries_kernel(nf3, fwkerhalf3, plan->spopts);
    }

    // copy to device 
    checkCudaErrors(cudaMemcpy(plan->fwkerhalf1,fwkerhalf1,(nf1/2+1)*
		sizeof(PCS),cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMemcpy(plan->fwkerhalf2,fwkerhalf2,(nf2/2+1)*
		sizeof(PCS),cudaMemcpyHostToDevice));
	if(w_term_method)
		checkCudaErrors(cudaMemcpy(plan->fwkerhalf3,fwkerhalf3,(nf3/2+1)*
			sizeof(PCS),cudaMemcpyHostToDevice));
    

    // cufft plan setting
    cufftHandle fftplan;
    int n[] = {N2, N1};
    int inembed[] = {plan->nf2, plan->nf1};
	int onembed[] = {N2, N1};
    
    // check, multi cufft for different w ??? how to set
	// cufftCreate(&fftplan);
	// cufftPlan2d(&fftplan,n[0],n[1],CUFFT_TYPE);
    // the bach size sets as the num of w when memory is sufficent. Alternative way, set as a smaller number when memory is insufficient.
    // and handle this piece by piece 
	cufftPlanMany(&fftplan,2,n,inembed,1,inembed[0]*inembed[1],
		onembed,1,onembed[0]*onembed[1],CUFFT_TYPE,plan->num_w); //need to check and revise (the partial conv will be differnt)
    plan->fftplan = fftplan; 
    

    // set up bin size +++ (for other methods) and related malloc based on gpu method
    // assign memory for index after sorting (can be done in setup_plan)
    // bin sorting (for other methods)
   


    // free host fwkerhalf
    free(fwkerhalf1);
    free(fwkerhalf2);
    if(w_term_method)free(fwkerhalf3);

    return ier;
}


int gridder_exectuion(curafft_plan* plan){
    /*
    Execute conv, fft, dft, correction for different direction (gridding or degridding)
    */
    int ier=0;
    // Mult-GPU support: set the CUDA Device ID:
        // int orig_gpu_device_id;
        // cudaGetDevice(& orig_gpu_device_id);
        // cudaSetDevice(d_plan->opts.gpu_device_id);

	int direction = plan->copts.direction;
    if (direction == 1){
        ier = exec_inverse(plan);
    }
    else{
        // forward not implement yet
        ier = 0;
    }
	

    // Multi-GPU support: reset the device ID
    // cudaSetDevice(orig_gpu_device_id);
    return ier;
}

int gridder_destroy(){
    // free memory
    int ier=0;
    return ier;
}