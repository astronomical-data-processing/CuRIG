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
#include "precomp.h"
#include "ra_exec.h"
#include "utils.h"
#include "cufft.h"
#include "deconv.h"
#include "cugridder.h"

int setup_gridder_plan(int N1, int N2, PCS fov, int lshift, int mshift, int nrow, PCS *d_w, CUCPX *d_c, conv_opts copts, ragridder_plan *gridder_plan, curafft_plan *plan)
{
    gridder_plan->fov = fov;
    gridder_plan->width = N1;
    gridder_plan->height = N2;
    gridder_plan->nrow = nrow;
    // determain number of w
    // ignore shift
    gridder_plan->pixelsize_x = fov / 180.0 * PI / (PCS)N2;
    gridder_plan->pixelsize_y = fov / 180.0 * PI / (PCS)N1;
    PCS xpixelsize = gridder_plan->pixelsize_x;
    PCS ypixelsize = gridder_plan->pixelsize_y;
    PCS l_min = lshift - 0.5 * xpixelsize * N2;
    PCS l_max = l_min + xpixelsize * (N2 - 1);

    PCS m_min = mshift - 0.5 * ypixelsize * N1;
    PCS m_max = m_min + ypixelsize * (N1 - 1);

    //double upsampling_fac = copts.upsampfac;
    PCS n_lm = sqrt(1.0 - pow(l_min, 2) - pow(m_min, 2));
    //printf("lmin lmax mmin mmax nlm, %lf, %lf, %lf, %lf, %lf\n",l_min,l_max,m_min,m_max,n_lm);
    // nshift = (no_nshift||(!do_wgridding)) ? 0. : -0.5*(nm1max+nm1min);
    PCS i_max, i_min;
    PCS o_min;
    get_max_min(i_max, i_min, d_w, gridder_plan->nrow);

    //PCS u_max = max(abs(i_max),abs(i_min));
    plan->ta.i_center[0] = (i_max + i_min) / (PCS)2.0;
    plan->ta.i_half_width[0] = (i_max - i_min) / (PCS)2.0;

    // conside all w and n-1
    // get_max_min(o_max, o_min, gridder_plan->d_x, N1);
    // printf("max min %lf, %lf\n",o_max,o_min);
    
    o_min = n_lm-1;
    plan->ta.o_center[0] =  o_min / (PCS)2.0;
    plan->ta.o_half_width[0] = abs(o_min / (PCS)2.0);
    //PCS x_max = max(abs(o_max),abs(o_min));

    set_nhg_type3(plan->ta.o_half_width[0], plan->ta.i_half_width[0], plan->copts, plan->nf1, plan->ta.h[0], plan->ta.gamma[0]); //temporately use nf1
    printf("U_width %lf, U_center %lf, X_width %.10lf, X_center %.10lf, gamma %lf, nf %d, h %lf\n",
           plan->ta.i_half_width[0], plan->ta.i_center[0], plan->ta.o_half_width[0], plan->ta.o_center[0], plan->ta.gamma[0], plan->nf1, plan->ta.h[0]);
    // u_j to u_j' x_k to x_k' c_j to c_j'
    checkCudaErrors(cudaMalloc((void **)&plan->d_x, sizeof(PCS) * (N1 / 2 + 1) * (N2 / 2 + 1)));
    w_term_k_generation(plan->d_x, N1, N2, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y);

    pre_stage_invoker(plan->ta.i_center, plan->ta.o_center, plan->ta.gamma, plan->ta.h, d_w, NULL, NULL, plan->d_x, NULL, NULL, d_c, gridder_plan->nrow,(N1 / 2 + 1) * (N2 / 2 + 1), 1, 1, plan->iflag);
    gridder_plan->num_w = plan->nf1;
    // PCS max, min;
    // PCS delta_w = 1 / (2 * upsampling_fac * abs(n_lm - 1));
    // gridder_plan->delta_w = delta_w;
    // get_max_min(max, min, d_w, gridder_plan->nrow);

    // gridder_plan->w_max = max;
    // gridder_plan->w_min = min;
    // gridder_plan->w_s_r = 1;
    // PCS w_0 = gridder_plan->w_min - delta_w * (copts.kw - 1); // first gridder_plane
    // gridder_plan->w_0 = w_0;
    // gridder_plan->num_w = ((gridder_plan->w_max - gridder_plan->w_min) / delta_w + copts.kw); // another plan

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
        direction - 1 vis to image, 0 currently not support
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

    // fov and other astro related setting +++

    // get effective coordinates: *1/lambda
    PCS f_over_c = pointer_v->frequency[0]/SPEEDOFLIGHT;
    printf("foverc %lf\n",f_over_c);
   

    get_effective_coordinate_invoker(d_u,d_v,d_w,f_over_c,pointer_v->pirange,M);

    // PCS *w = (PCS *) malloc(sizeof(PCS)*M);
    // checkCudaErrors(cudaMemcpy(w,d_w,sizeof(PCS)*M,cudaMemcpyDeviceToHost));
   
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

    ier = setup_conv_opts(plan->copts, tol, sigma, 1, direction, kerevalmeth); //check the arguements pirange = 1

    int fftsign = (iflag >= 0) ? 1 : -1;
    plan->iflag = fftsign; //may be useless| conflict with direction
    if (ier != 0)
        printf("setup_error\n");
    
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
    setup_gridder_plan(N1, N2, fov, 0, 0, M, d_w, d_c, plan->copts, gridder_plan, plan);
    
    // gridder_plan->num_w = 80;
    int nf1 = get_num_cells(N1, plan->copts);
    int nf2 = get_num_cells(N2, plan->copts);
    int nf3 = gridder_plan->num_w;

    if (w_term_method)
        plan->dim = 3;
    else
        plan->dim = 2;
    setup_plan(nf1, nf2, nf3, M, d_v, d_u, d_w, d_c, plan);

    // printf("input data checking cugridder...\n");
    //         PCS *temp = (PCS*)malloc(sizeof(PCS)*10);
    //         printf("u v w and vis\n");
    //         cudaMemcpy(temp,d_u,sizeof(PCS)*10,cudaMemcpyDeviceToHost);
    //         for(int i=0;i<10;i++)
    //         printf("%.3lf ",temp[i]);
    //         printf("\n");

    plan->ms = N1;
    plan->mt = N2;
    plan->mu = 1;
    plan->execute_flow = 1;
    plan->fw = NULL; //allocated in precomp
    
    batchsize = gridder_plan->num_w;
    plan->batchsize = batchsize;

    // plan->copts.direction = direction; // 1 inverse, 0 forward

    // // fw allocation
    // checkCudaErrors(cudaMalloc((void**)&plan->fw,sizeof(CUCPX)*nf1*nf2*nf3));

    fourier_series_appro_invoker(plan->fwkerhalf1, plan->copts, plan->nf1 / 2 + 1);
    fourier_series_appro_invoker(plan->fwkerhalf2, plan->copts, plan->nf2 / 2 + 1);

    if (w_term_method)
    {
        // improved_ws
        checkCudaErrors(cudaFree(plan->fwkerhalf3));
        checkCudaErrors(cudaMalloc((void **)&plan->fwkerhalf3, sizeof(PCS) * (N1 / 2 + 1) * (N2 / 2 + 1)));
        //PCS *k;
        //w_term_k_generation(k, plan->nf1, plan->nf2, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y, gridder_plan->w_s_r);
        fourier_series_appro_invoker(plan->fwkerhalf3, plan->d_x, plan->copts, (N1 / 2 + 1) * (N2 / 2 + 1)); // correction with k, may be wrong, k will be free in this function
    }

    PCS *fwkerhalf1 = (PCS *)malloc(sizeof(PCS) * (plan->nf1 / 2 + 1));
    PCS *fwkerhalf2 = (PCS *)malloc(sizeof(PCS) * (plan->nf2 / 2 + 1));

    cudaMemcpy(fwkerhalf1, plan->fwkerhalf1, sizeof(PCS) * (plan->nf1 / 2 + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(fwkerhalf2, plan->fwkerhalf2, sizeof(PCS) * (plan->nf2 / 2 + 1), cudaMemcpyDeviceToHost);

    // printf("correction factor printing...\n");
    // for(int i=0; i<plan->nf1/2+1; i++) printf("%lf ",fwkerhalf1[i]); printf("\n");
    // for(int i=0; i<plan->nf2/2+1; i++) printf("%lf ",fwkerhalf2[i]); printf("\n");

    // cufft plan setting
    cufftHandle fftplan;
    int n[] = {plan->nf2, plan->nf1};
    int inembed[] = {plan->nf2, plan->nf1};
    int onembed[] = {plan->nf2, plan->nf1};

    // check, multi cufft for different w ??? how to set
    // cufftCreate(&fftplan);
    // cufftPlan2d(&fftplan,n[0],n[1],CUFFT_TYPE);
    // the bach size sets as the num of w when memory is sufficent. Alternative way, set as a smaller number when memory is insufficient.
    // and handle this piece by piece
    cufftPlanMany(&fftplan, 2, n, inembed, 1, inembed[0] * inembed[1],
                  onembed, 1, onembed[0] * onembed[1], CUFFT_TYPE, plan->nf3); //need to check and revise (the partial conv will be differnt)
    plan->fftplan = fftplan;


    // u and v scaling *pixelsize
    rescaling_real_invoker(d_u,gridder_plan->pixelsize_x,gridder_plan->nrow);
    rescaling_real_invoker(d_v,gridder_plan->pixelsize_y,gridder_plan->nrow);

    // fw malloc
    checkCudaErrors(cudaMalloc((void**)&plan->fw,sizeof(CUCPX)*plan->nf1*plan->nf2*plan->nf3));
    checkCudaErrors(cudaMemset(plan->fw, 0, plan->nf3 * plan->nf1 * plan->nf2 * sizeof(CUCPX)));

    // set up bin size +++ (for other methods) and related malloc based on gpu method
    // assign memory for index after sorting (can be done in setup_plan)
    // bin sorting (for other methods)

    // free host fwkerhalf
    // free(fwkerhalf1);
    // free(fwkerhalf2);
    // if(w_term_method)free(fwkerhalf3);

    return ier;
}

int gridder_execution(curafft_plan *plan, ragridder_plan *gridder_plan)
{
    /*
    Execute conv, fft, dft, correction for different direction (gridding or degridding)
    */
    int ier = 0;
    // Mult-GPU support: set the CUDA Device ID:
    // int orig_gpu_device_id;
    // cudaGetDevice(& orig_gpu_device_id);
    // cudaSetDevice(d_plan->opts.gpu_device_id);
    int direction = plan->copts.direction;

    if (direction == 1)
    {
        ier = exec_vis2dirty(plan, gridder_plan);
    }
    else
    {
        // forward not implement yet
        ier = 0;
    }

    // Multi-GPU support: reset the device ID
    // cudaSetDevice(orig_gpu_device_id);
    return ier;
}

int gridder_destroy(curafft_plan *plan, ragridder_plan *gridder_plan)
{
    // free memory
    int ier = 0;
    checkCudaErrors(cudaFree(plan->d_x));
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