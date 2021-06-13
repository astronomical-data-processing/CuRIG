/*
Some precomputation related radio astronomy
    Sampling
    Vis * weight
    ...
*/
#include "dataType.h"
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

__global__ void gridder_rescaling_complex(CUCPX *x, PCS scale_ratio, int N){
    int idx;
    for(idx = blockIdx.x * blockDim.x; idx<N; idx += gridDim.x * blockDim.x){
        x[idx] *= scale_ratio;
    }
}

__global__ void gridder_rescaling_real(PCS *x, PCS scale_ratio, int N){
    int idx;
    for(idx = blockIdx.x * blockDim.x; idx<N; idx += gridDim.x * blockDim.x){
        x[idx] *= scale_ratio;
    }
}

void pre_setting(PCS *d_u, PCS *d_v, PCS *d_w, PCS d_vis, ragridder_plan *gridder_plan){
    PCS f_over_c = gridder_plan->kv.frequency[gridder_plan->cur_channel]/SPEEDOFLIGHT;
    PCS xpixelsize = gridder_plan->pixelsize_x;
    PCS ypixelsize = gridder_plan->pixelsize_y;
    int pirange = gridder_plan->kv.pirange;
    int nrow = gridder_plan->nrow;
    int N = nrow;
    int blocksize = 512;
    // ---------get effective coordinates---------
    get_effective_coordinate<<<(N-1)/blocksize+1, blocksize>>(d_u, d_v, d_w, f_over_c, pirange, nrow);
    cudaCheckErrors(cudaDeviceSynchronize());
    // ----------------rescaling-----------------
    PCS scaling_ratio = 1.0/xpixelsize;
    gridder_rescaling_real<<<(N-1)/blocksize+1, blocksize>>>(d_u, scaling_ratio, nrow);
    cudaCheckErrors(cudaDeviceSynchronize());
    scaling_ratio = 1.0/ypixelsize;
    gridder_rescaling_real<<<(N-1)/blocksize+1, blocksize>>>(d_v, scaling_ratio, nrow);
    cudaCheckErrors(cudaDeviceSynchronize());
    // ------------vis * flag * weight--------+++++
    // memory transfer (vis belong to this channel and weight)
	checkCudaErrors(cudaMemcpy(d_vis, vis+nrow*gridder_plan->cur_channel, nrow * sizeof(CUCPX), cudaMemcpyHostToDevice)); //
}

__global__ void explicit_gridder(int N1, int N2, int nrow, PCS *u, PCS *v, PCS *w, CUCPX *vis, 
        CUCPX *dirty, PCS f_over_c, PCS row_pix_size, PCS col_pix_size){
    /*
        N1,N2 - width, height 
        row_pix_size, col_pix_size - xpixsize, ypixsize
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row;
    int col;
    PCS l, m, n_lm;
    CUCPX res(0,0);
    CUCPX temp;
    for(idx=0; idx<N1 * N2; idx+=gridDim.x * blockDim.x){
        row = idx / N1 - int(0.5*N2);
        col = idx % N1 - int(0.5*N1);
        l = row * row_pix_size;
        m = col * col_pix_size;
        n_lm = sqrt(1 - l**2 - m**2);
        for(int i=0; i<nrow; i++){
            PCS phase = f_over_c*(l*u[i] + m*v[i] + (n_lm-1)*w[i]);
            if(pirange != 1) phase = phase * 2 * PI;
            temp.x = cos(phase),
            temp.y = sin(phase);
            res += vis[i] * temp; 
        }
        dirty[idx] += res/n_ml; // add values of all channels
    }
}

void explicit_gridder_invoker(ragridder_plan *gridder_plan){
    int nchan = gridder_plan->channel;
    int nrow = gridder_plan->nrow;
    int N1 = gridder_plan->width;
    int N2 = gridder_plan->height;
    PCS *d_u, *d_v, *d_w;
    CUCPX *d_vis, *d_dirty;
    cudaCheckErrors(cudaMalloc((void **)&d_u, sizeof(PCS)*nrow));
    cudaCheckErrors(cudaMalloc((void **)&d_v, sizeof(PCS)*nrow));
    cudaCheckErrors(cudaMalloc((void **)&d_w, sizeof(PCS)*nrow));
    cudaCheckErrors(cudaMalloc((void)&d_vis, sizeof(CUCPX)*nrow));
    cudaCheckErrors(cudaMalloc((void)&d_dirty, sizeof(CUCPX)*nrow));

    cudaCheckErrors(cudaMemcpy(d_u, gridder_plan->kv.u, sizeof(PCS)*nrow, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_v, gridder_plan->kv.v, sizeof(PCS)*nrow, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_w, gridder_plan->kv.w, sizeof(PCS)*nrow, cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpy(d_vis, gridder_plan->kv.vis, sizeof(CUCPX)*nrow, cudaMemcpyHostToDevice));
    
    int blocksize = 1024;
    PCS f_over_c;
    for(int i=0; i<nchan; i++){
        cudaCheckErrors(cudaMemcpy(d_vis, gridder_plan->kv.vis+i*nrow, sizeof(CUCPX)*nrow, cudaMemcpyHostToDevice));
        f_over_c = gridder_plan->kv.frequency[i]/SPEEDOFLIGHT;
        explicit_gridder<<<(N1*N2-1)/blocksize+1, blocksize>>>(N1, N2, nrow, d_u, d_v, d_w, d_vis, 
        d_dirty, f_over_c, xpixsize, ypixsize);
        cudaCheckErrors(cudaDeviceSynchronize());
    }
    cudaCheckErrors(cudaMemcpy(gridder_plan->dirty_image, d_dirty, sizeof(PCS)*nrow, cudaMemcpyDeviceToHost));
    cudaCheckErrors(cudaFree(d_u));
    cudaCheckErrors(cudaFree(d_v));
    cudaCheckErrors(cudaFree(d_w));
    cudaCheckErrors(cudaFree(d_vis));
    cudaCheckErrors(cudaFree(d_dirty));
}