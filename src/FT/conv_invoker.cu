/*
Invoke conv related kernel
*/

//#include "../memtransfer.h"
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <cuComplex.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include "dataType.h"
#include "conv_invoker.h"
#include "conv.h"

/*
int setup_conv_opts(conv_opts *c_opts, PCS eps, PCS upsampfac, int kerevalmeth){
  int ier = 0;
  kerevalmeth = 1;
  return ier;
}
*/

/*
void get_max_min(PCS *x, int num, PCS *h_res){
  PCS *d_res;
  CHECK(cudaMalloc((void **)&d_res,sizeof(PCS)*2));
  reduce_max_min<<<(num-1)/BLOCKSIZE+1,BLOCKSIZE>>>(x,num,d_res);
  CHECK(cudaDeviceS...);
  CHECK(cudaMemcpy(h_res,d_res,sizeof(PCS)*2),cudaMemcpyDeviceToHost);
  cudaFree(d_res);
}
*/

int setup_plan(int nf1, int nf2, int M, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, curafft_plan *plan)
{
  /* different dim will have different setting
    ----plan setting, and related memory allocation----
        nf1, nf2 - number of UPTS (resolution of image)
        M - number of NUPTS (num of vis)
        d_u, d_v, d_w - location
        d_c - value
    */
  int ier = 0;
  //wrong here
  /*
  plan->kv.u = d_u;
  plan->kv.v = d_v;
  plan->kv.w = d_w;
  plan->kv.vis = d_c;
  */ 
  //int ier;
  plan->nf1 = nf1;
  plan->nf2 = nf2;

  /*
  //get number of w
  int num_w = 0;
  //reduce to get maximum and minimum, h_res[0] max, [1] min
  PCS *h_res = (PCS *)malloc(sizeof(int)*2);
  PCS max = h_res[0];
  PCS min = h_res[1];
  free(h_res);
  PCS n_scale = sqrt(max(1. - l_max * l_max - m_max * m_max, 0.)) - 1.;
  if (l_max * l_max + m_max * m_max > 1.)
    n_scale = -sqrt(abs(1. - l_max * l_max - m_max * m_max)) - 1.;
  plan->num_w =  abs(n_scale)/(0.25) * (max-min) + plan->copts.kw;
  */
  plan->num_w = 2 * nf1;

  plan->M = M;
  //plan->maxbatchsize = 1;

  plan->byte_now = 0;
  // No extra memory is needed in nuptsdriven method (case 1)
  switch (plan->opts.gpu_method)
  {
  case 0:
  {
    if (plan->opts.gpu_sort)
    {
      CHECK(cudaMalloc(&plan->cell_loc, sizeof(INT_M) * M));
    }
  }
  case 1:
  {
    if (plan->opts.gpu_sort)
    {
      CHECK(cudaMalloc(&plan->cell_loc, sizeof(INT_M) * M));
    }
  }
  break;
  
  default:
    std::cerr << "err: invalid method " << std::endl;
  }
  return ier;
}

  int ws_conv(int nf1, int nf2, int nf3, int M, curafft_plan *plan)
  {
    return 0;
  }

  int improved_ws_conv(int nf1, int nf2, int nf3, int M, curafft_plan *plan)
  {
    //add content

    dim3 grid;
    dim3 block;
    if (plan->opts.gpu_method == 0)
    {
      block.x = 256;
      grid.x = (M - 1) / block.x + 1;
      conv_3d_nputsdriven<<<grid, block>>>(plan->kv.u, plan->kv.v, plan->kv.w, plan->kv.vis, plan->fw, plan->M,
                                           plan->copts.kw, nf1, nf2, nf3, plan->copts.ES_c, plan->copts.ES_beta, plan->copts.pirange, plan->cell_loc);
    }

    return 0;
  }

  int curafft_conv(curafft_plan * plan)
  {
    /*
    ---- convolution opertion ----
    */

    int ier = 0;
    int nf1 = plan->nf1;
    int nf2 = plan->nf2;
    int nf3 = plan->num_w;
    int M = plan->M;
    if (plan->w_term_method == 0)
    {
      ws_conv(nf1, nf2, nf3, M, plan);
    }
    if (plan->w_term_method == 1)
    {
      //get nupts location in grid cells
      improved_ws_conv(nf1, nf2, nf3, M, plan);
    }
    return ier;
  }
