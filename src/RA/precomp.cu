/*
Some precomputation related radio astronomy
    Sampling
    Vis * weight
    ...
*/
#include "dataType.h"
#include "curafft_plan.h"
#include "ragridder_plan.h"
#include "utils.h"
#include "precomp.h"

__global__ void get_effective_coordinate(PCS *u, PCS *v, PCS *w, PCS f_over_c, int pirange,int nrow){
    /*
        u, v, w - coordinate
        f_over_c - frequency divide speed of light
        pirange - 1 in [-pi,pi), 0 - [-0.5,0.5)
        nrow - number of coordinates
    */
    int idx;
    for(idx = blockDim.x * blockIdx.x + threadIdx.x; idx<nrow; idx+= gridDim.x * blockDim.x){
        u[idx] *= f_over_c;
        v[idx] *= f_over_c;
        u[idx] *= f_over_c;
        if(!pirange){
            u[idx] *= PI;
            v[idx] *= PI;
            u[idx] *= PI;
        }
    }
}


__global__ void gridder_rescaling_real(PCS *x, PCS scale_ratio, int N){
    int idx;
    for(idx = blockIdx.x * blockDim.x; idx<N; idx += gridDim.x * blockDim.x){
        x[idx] *= scale_ratio;
    }
}

void pre_setting(PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_vis, curafft_plan *plan, ragridder_plan *gridder_plan){
    PCS f_over_c = gridder_plan->kv.frequency[gridder_plan->cur_channel]/SPEEDOFLIGHT;
    PCS xpixelsize = gridder_plan->pixelsize_x;
    PCS ypixelsize = gridder_plan->pixelsize_y;
    int pirange = gridder_plan->kv.pirange;
    int nrow = gridder_plan->nrow;
    int N = nrow;
    int blocksize = 512;
    //----------w max min num_w w_0--------------
    gridder_plan->w_max *= f_over_c;
    gridder_plan->w_min *= f_over_c;
    PCS w_0 = gridder_plan->w_min - gridder_plan->delta_w * (plan->copts.kw - 1); // first plane
    gridder_plan->w_0 = w_0;
    int num_w = (gridder_plan->w_max - gridder_plan->w_min)/gridder_plan->delta_w + plan->copts.kw;
    gridder_plan->num_w = num_w;
    if(gridder_plan->cur_channel!=0){
        gridder_plan->w_max /= gridder_plan->kv.frequency[gridder_plan->cur_channel-1]/SPEEDOFLIGHT;
        gridder_plan->w_min /= gridder_plan->kv.frequency[gridder_plan->cur_channel-1]/SPEEDOFLIGHT;
    }
    //----------plan reset--------------
    plan->nf3 = num_w;
    plan->batchsize = min(4,num_w);
    if(gridder_plan->w_term_method){
		// improved_ws
        checkCudaErrors(cudaFree(plan->fwkerhalf3));
        PCS *fwkerhalf3 = (PCS*)malloc(sizeof(PCS)*(plan->nf3/2+1));
        //need to revise
        onedim_fseries_kernel(gridder_plan->num_w, fwkerhalf3, plan->copts);
        checkCudaErrors(cudaMalloc((void**)&plan->fwkerhalf3,sizeof(PCS)*(plan->nf3/2+1)));
        checkCudaErrors(cudaMemcpy(plan->fwkerhalf3,fwkerhalf3,(plan->nf3/2+1)*
			sizeof(PCS),cudaMemcpyHostToDevice));
        free(fwkerhalf3);
        //fw
        if(plan->fw!=NULL)checkCudaErrors(cudaFree(plan->fw));
        checkCudaErrors(cudaMalloc((void**)&plan->fw,sizeof(CUCPX)*plan->nf1*plan->nf2*plan->nf3));
        checkCudaErrors(cudaMemset(plan->fw, 0, plan->nf3 * plan->nf1 * plan->nf2 * sizeof(CUCPX)));
    }
    int N1 = plan->ms;
    int N2 = plan->mt;
    int n[] = {N2, N1};
    int inembed[] = {plan->nf2, plan->nf1};
	int onembed[] = {N2, N1};
    cufftPlanMany(&plan->fftplan,2,n,inembed,1,inembed[0]*inembed[1],
		onembed,1,onembed[0]*onembed[1],CUFFT_TYPE,plan->nf3);
    // ---------get effective coordinates---------
    get_effective_coordinate<<<(N-1)/blocksize+1, blocksize>>>(d_u, d_v, d_w, f_over_c, pirange, nrow);
    checkCudaErrors(cudaDeviceSynchronize());
    // ----------------rescaling-----------------
    PCS scaling_ratio = 1.0/xpixelsize;
    gridder_rescaling_real<<<(N-1)/blocksize+1, blocksize>>>(d_u, scaling_ratio, nrow);
    checkCudaErrors(cudaDeviceSynchronize());
    scaling_ratio = 1.0/ypixelsize;
    gridder_rescaling_real<<<(N-1)/blocksize+1, blocksize>>>(d_v, scaling_ratio, nrow);
    checkCudaErrors(cudaDeviceSynchronize());
    // ------------vis * flag * weight--------+++++
    // memory transfer (vis belong to this channel and weight)
	checkCudaErrors(cudaMemcpy(d_vis, gridder_plan->kv.vis + nrow*gridder_plan->cur_channel, nrow * sizeof(CUCPX), cudaMemcpyHostToDevice)); //
}

__global__ void explicit_gridder(int N1, int N2, int nrow, PCS *u, PCS *v, PCS *w, CUCPX *vis, 
        CUCPX *dirty, PCS f_over_c, PCS row_pix_size, PCS col_pix_size, int pirange){
    /*
        N1,N2 - width, height 
        row_pix_size, col_pix_size - xpixsize, ypixsize
    */
    int idx;
    int row;
    int col;
    PCS l, m, n_lm;
    CUCPX temp;
    for(idx=blockIdx.x * blockDim.x + threadIdx.x; idx<N1 * N2; idx+=gridDim.x * blockDim.x){
        temp.x = 0.0; temp.y = 0.0;
        row = idx / N1 - (int)(0.5*N2);
        col = idx % N1 - (int)(0.5*N1);
        l = row * row_pix_size;
        m = col * col_pix_size;
        n_lm = sqrt(1 - pow(l,2) - pow(m,2));
        for(int i=0; i<nrow; i++){
            PCS phase = f_over_c*(l*u[i] + m*v[i] + (n_lm-1)*w[i]);
            if(pirange != 1) phase = phase * 2 * PI;
            temp.x += vis[i].x * cos(phase) - vis[i].y * sin(phase);
            temp.y += vis[i].x * sin(phase) + vis[i].y * cos(phase);
            printf("temp.x .y %lf %lf\n",temp.x,temp.y);
        }
    }
}

void explicit_gridder_invoker(ragridder_plan *gridder_plan){
    int nchan = gridder_plan->channel;
    int nrow = gridder_plan->nrow;
    int N1 = gridder_plan->width;
    int N2 = gridder_plan->height;
    int pirange = gridder_plan->kv.pirange;
    PCS xpixsize = gridder_plan->pixelsize_x;
    PCS ypixsize = gridder_plan->pixelsize_y;
    PCS *d_u, *d_v, *d_w;
    CUCPX *d_vis, *d_dirty;
    checkCudaErrors(cudaMalloc((void **)&d_u, sizeof(PCS)*nrow));
    checkCudaErrors(cudaMalloc((void **)&d_v, sizeof(PCS)*nrow));
    checkCudaErrors(cudaMalloc((void **)&d_w, sizeof(PCS)*nrow));
    checkCudaErrors(cudaMalloc((void **)&d_vis, sizeof(CUCPX)*nrow));
    checkCudaErrors(cudaMalloc((void **)&d_dirty, sizeof(CUCPX)*N1*N2));
    checkCudaErrors(cudaMemset(d_dirty, 0, sizeof(CUCPX)*N1*N2));

    checkCudaErrors(cudaMemcpy(d_u, gridder_plan->kv.u, sizeof(PCS)*nrow, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_v, gridder_plan->kv.v, sizeof(PCS)*nrow, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_w, gridder_plan->kv.w, sizeof(PCS)*nrow, cudaMemcpyHostToDevice));
    
    int blocksize = 1024;
    PCS f_over_c;
    for(int i=0; i<nchan; i++){
        checkCudaErrors(cudaMemcpy(d_vis, gridder_plan->kv.vis+i*nrow, sizeof(CUCPX)*nrow, cudaMemcpyHostToDevice));
        f_over_c = gridder_plan->kv.frequency[i]/SPEEDOFLIGHT;
        explicit_gridder<<<(N1*N2-1)/blocksize+1, blocksize>>>(N1, N2, nrow, d_u, d_v, d_w, d_vis, 
        d_dirty, f_over_c, xpixsize, ypixsize, pirange);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    checkCudaErrors(cudaMemcpy(gridder_plan->dirty_image, d_dirty, sizeof(CUCPX)*N1*N2, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_u));
    checkCudaErrors(cudaFree(d_v));
    checkCudaErrors(cudaFree(d_w));
    checkCudaErrors(cudaFree(d_vis));
    checkCudaErrors(cudaFree(d_dirty));
}