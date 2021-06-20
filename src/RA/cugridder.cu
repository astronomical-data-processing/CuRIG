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
#include "ragridder_plan.h"
#include "curafft_plan.h"
#include "cuft.h"
#include "ra_exec.h"
#include "utils.h"
#include "cugridder.h"


int setup_gridder_plan(int N1, int N2, PCS fov, int lshift, int mshift, int nrow, PCS *d_w, conv_opts copts, ragridder_plan *plan){
    plan->fov = fov;
    plan->width = N1;
    plan->height = N2;
    plan->nrow = nrow;
    // determain number of w 
    // ignore shift
    plan->pixelsize_x = fov / 180.0 * PI / (PCS)N1;
    plan->pixelsize_y = fov / 180.0 * PI / (PCS)N2;
    PCS xpixelsize = plan->pixelsize_x;
    PCS ypixelsize = plan->pixelsize_y;
    PCS l_min = lshift - 0.5*xpixelsize * N1;
    PCS l_max = l_min + xpixelsize * (N1-1);
    
    PCS m_min = mshift - 0.5*ypixelsize * N2;
    PCS m_max = m_min + ypixelsize * (N2-1);

    double upsampling_fac = copts.upsampfac;
    PCS n_lm = sqrt(1 - pow(l_max,2) + pow(m_max, 2));
    // nshift = (no_nshift||(!do_wgridding)) ? 0. : -0.5*(nm1max+nm1min);
    PCS max, min;
    PCS delta_w = 1/(2*upsampling_fac*abs(n_lm-1));
    plan->delta_w = delta_w;
    get_max_min(max, min, d_w, plan->nrow);

    plan->w_max = max;
    plan->w_min = min;
    PCS w_0 = plan->w_min - delta_w * (copts.kw - 1); // first plane
    plan->w_0 = w_0;
    plan->num_w = (plan->w_max - plan->w_min)/delta_w + copts.kw; // another plan

    return 0;
}

// the bin sort should be completed at gridder_settting

int gridder_setting(int N1, int N2, int method, int kerevalmeth, int w_term_method, PCS tol, int direction, double sigma, int iflag,
    int batchsize, int M, int channel, PCS fov, visibility *pointer_v, PCS *d_u, PCS *d_v, PCS *d_w,
    CUCPX *d_c, curafft_plan *plan, ragridder_plan *gridder_plan)
{
    /*
        N1, N2 - number of Fouier modes
        method - gridding method
        kerevalmeth - gridding kernel evaluation method
        tol - tolerance (epsilon)
        direction - 1 CUFFT_INVERSE, 0 CUFFT_FORWARD
        sigma - upsampling factor
        iflag - flag for fourier transform
        batchsize - number of batch in  cufft (used for handling piece by piece)
        M - number of nputs (visibility)
        channel - number of channels
        wgt - weight
        freq - frequency
        d_u, d_v, d_w - wavelengths in different dimensions, x is on host, d_x is on device
        d_c - value of visibility

        ****issue, degridding
    */
    int ier = 0;
    
    plan = new curafft_plan();
    gridder_plan = new ragridder_plan();
    memset(plan, 0, sizeof(curafft_plan));
    memset(gridder_plan, 0, sizeof(ragridder_plan));

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

    ier = setup_conv_opts(plan->copts, tol, sigma, 0, direction, kerevalmeth); //check the arguements

	if(ier!=0)printf("setup_error\n");

    // plan setting
    // cuda stream malloc in setup_plan
    gridder_plan->channel = channel;
    gridder_plan->w_term_method = w_term_method;
    gridder_plan->speedoflight = SPEEDOFLIGHT;
    gridder_plan->kv.u = pointer_v->u;
    gridder_plan->kv.v = pointer_v->v;
    gridder_plan->kv.w = pointer_v->w;
    gridder_plan->kv.vis = pointer_v->vis;
    gridder_plan->kv.weight = pointer_v->weight;
    gridder_plan->kv.frequency = pointer_v->frequency;
    gridder_plan->kv.pirange = pointer_v->pirange;
    setup_gridder_plan(N1,N2,fov,0,0,M,d_w,plan->copts,gridder_plan);
    

    int nf1 = get_num_cells(N1,plan->copts);
    int nf2 = get_num_cells(N2,plan->copts);
    int nf3 = gridder_plan->num_w;
    
    if(w_term_method)plan->dim =3;
    else plan->dim = 2;
    setup_plan(nf1, nf2, nf3, M, d_u, d_v, d_w, d_c, plan);
    
    // plan->dim = dim;
	plan->ms = N1;
	plan->mt = N2;
	// plan->mu = nmodes[2];

    int fftsign = (iflag>=0) ? 1 : -1;

	plan->iflag = fftsign;
    if (batchsize == 0) batchsize = min(4,gridder_plan->num_w);
	plan->batchsize = batchsize;

    plan->copts.direction = direction; // 1 inverse, 0 forward

    // // fw allocation
    // checkCudaErrors(cudaMalloc((void**)&plan->fw,sizeof(CUCPX)*nf1*nf2*nf3));

    PCS *fwkerhalf1 = (PCS*)malloc(sizeof(PCS)*(plan->nf1/2+1));
    onedim_fseries_kernel(plan->nf1, fwkerhalf1, plan->copts); // used for correction
    
    PCS *fwkerhalf2 = (PCS*)malloc(sizeof(PCS)*(plan->nf2/2+1));
    onedim_fseries_kernel(plan->nf2, fwkerhalf2, plan->copts);

    // copy to device 
    checkCudaErrors(cudaMemcpy(plan->fwkerhalf1,fwkerhalf1,(plan->nf1/2+1)*
		sizeof(PCS),cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMemcpy(plan->fwkerhalf2,fwkerhalf2,(plan->nf2/2+1)*
		sizeof(PCS),cudaMemcpyHostToDevice));
    PCS *fwkerhalf3;
	if(w_term_method){
		// improved_ws
        fwkerhalf3 = (PCS*)malloc(sizeof(PCS)*(plan->nf3/2+1));
        //need to revise
        onedim_fseries_kernel(gridder_plan->num_w, fwkerhalf3, plan->copts);
        checkCudaErrors(cudaMemcpy(plan->fwkerhalf3,fwkerhalf3,(gridder_plan->num_w/2+1)*
			sizeof(PCS),cudaMemcpyHostToDevice));
    }
    

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
		onembed,1,onembed[0]*onembed[1],CUFFT_TYPE,plan->nf3); //need to check and revise (the partial conv will be differnt)
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


int gridder_exectuion(curafft_plan* plan, ragridder_plan* gridder_plan){
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
        ier = exec_inverse(plan, gridder_plan);
    }
    else{
        // forward not implement yet
        ier = 0;
    }
	

    // Multi-GPU support: reset the device ID
    // cudaSetDevice(orig_gpu_device_id);
    return ier;
}

int gridder_destroy(curafft_plan *plan, ragridder_plan *gridder_plan){
    // free memory
    int ier=0;
    curafft_free(plan);
    free(plan);
    free(gridder_plan->dirty_image);
    free(gridder_plan->kv.u);
    free(gridder_plan->kv.v);
    free(gridder_plan->kv.w);
    free(gridder_plan->kv.vis);
    free(gridder_plan->kv.frequency);
    free(gridder_plan->kv.weight);
    // free(gridder_plan->kv.flag);
    free(gridder_plan);
    return ier;
}