/*
Invoke conv related kernel
Issue: Revise for batch
*/

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
#include "conv_invoker.h"
#include "conv.h"
#include "utils.h"

int get_num_cells(int ms, conv_opts copts)
// type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and requested number of Fourier modes ms.
{
  int nf = (int)(copts.upsampfac*ms);
  if (nf<2*copts.kw) nf=2*copts.kw; // otherwise spread fails
  if (nf<1e11){                                // otherwise will fail anyway
      nf = next235beven(nf, 1);
  }
  return nf;
}

int setup_conv_opts(conv_opts &opts, PCS eps, PCS upsampfac, int kerevalmeth)
{
  /*
    setup conv related components
  */
  if (upsampfac != 2.0)
  { // nonstandard sigma
    if (kerevalmeth == 1)
    {
      fprintf(stderr, "setup_conv_opts: nonstandard upsampfac with kerevalmeth=1\n", (double)upsampfac);
      return 2;
    }
    if (upsampfac <= 1.0)
    {
      fprintf(stderr, "setup_conv_opts: error, upsampling factor too small\n", (double)upsampfac);
      return 2;
    }
    // calling routine must abort on above errors, since opts is garbage!
    if (upsampfac > 4.0)
      fprintf(stderr, "setup_conv_opts: warning, upsampfac=%.3g is too large\n", (double)upsampfac);
  }

  // defaults... (user can change after this function called)
  opts.direction = 1; // user should always set to 1 or 2 as desired
  opts.pirange = 1;   // user also should always set this
  opts.upsampfac = upsampfac;

  // as in FINUFFT v2.0, allow too-small-eps by truncating to eps_mach...
  int ier = 0;
  if (eps < EPSILON)
  {
    fprintf(stderr, "setup_conv_opts: warning, eps (tol) is too small, set eps = %.3g.\n", (double)eps, (double)EPSILON);
    eps = EPSILON;
    ier = 1;
  }

  // Set kernel width w (aka kw) and ES kernel beta parameter, in opts...
  int kw = std::ceil(-log10(eps / (PCS)10.0));                  // 1 digit per power of ten
  if (upsampfac != 2.0)                                         // override ns for custom sigma
    kw = std::ceil(-log(eps) / (PI * sqrt(1 - 1 / upsampfac))); // formula, gamma=1
  kw = max(2, kw);                                              
  if (kw > MAX_KERNEL_WIDTH)
  { // clip to match allocated arrays
    fprintf(stderr, "%s warning: at upsampfac=%.3g, tol=%.3g would need kernel width ns=%d; clipping to max %d, better to revise sigma and tol.\n", __func__,
            upsampfac, (double)eps, kw, MAX_KERNEL_WIDTH);
    kw = MAX_KERNEL_WIDTH;
    ier = 1;
  }
  opts.kw = kw;
  opts.ES_halfwidth = (PCS)kw / 2; // constants to help ker eval (except Horner)
  opts.ES_c = 4.0 / (PCS)(kw * kw);

  PCS betaoverns = 2.30; // gives decent betas for default sigma=2.0
  if (kw == 2)
    betaoverns = 2.20; // some small-width tweaks...
  if (kw == 3)
    betaoverns = 2.26;
  if (kw == 4)
    betaoverns = 2.38;
  if (upsampfac != 2.0)
  {                                                      // again, override beta for custom sigma
    PCS gamma = 0.97;                                    // must match devel/gen_all_horner_C_code.m
    betaoverns = gamma * PI * (1 - 1 / (2 * upsampfac)); // formula based on cutoff
  }
  opts.ES_beta = betaoverns * (PCS)kw; // set the kernel beta parameter
  printf("the value of beta %.3f\n",opts.ES_beta);
  //fprintf(stderr,"setup_spreader: sigma=%.6f, chose ns=%d beta=%.6f\n",(double)upsampfac,ns,(double)opts.ES_beta); // user hasn't set debug yet
  return ier;
}

// int ws_conv(int nf1, int nf2, int nf3, int M, curafft_plan *plan)
// {
//   return 0;
// }

int conv_1d_invoker(int nf1, int M, curafft_plan *plan){
  dim3 grid;
  dim3 block;
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256;
    grid.x = (M - 1) / block.x + 1;

    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    conv_1d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, plan->copts.ES_c, plan->copts.ES_beta, plan->copts.pirange, plan->cell_loc);
    
    checkCudaErrors(cudaDeviceSynchronize());
  }
}

int conv_2d_invoker(int nf1, int nf2, int M, curafft_plan *plan)
{

  dim3 grid;
  dim3 block;
  // printf("gpu_method %d\n",plan->opts.gpu_method);
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256;
    grid.x = (M - 1) / block.x + 1;
    //for debug

    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    conv_2d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_v, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, nf2, plan->copts.ES_c, plan->copts.ES_beta, plan->copts.pirange, plan->cell_loc);
    

    checkCudaErrors(cudaDeviceSynchronize());
  }

  return 0;
}

int conv_3d_invoker(int nf1, int nf2, int nf3, int M, curafft_plan *plan)
{

  dim3 grid;
  dim3 block;
  // printf("gpu_method %d\n",plan->opts.gpu_method);
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256;
    grid.x = (M - 1) / block.x + 1;
    //for debug

    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    conv_3d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_v, plan->d_w, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, nf2, nf3, plan->copts.ES_c, plan->copts.ES_beta, plan->copts.pirange, plan->cell_loc);
    

    checkCudaErrors(cudaDeviceSynchronize());
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
  int nf3 = plan->nf3;
  int M = plan->M;
  // printf("w_term_method %d\n",plan->w_term_method);
  switch (plan->dim)
  {
  case 1:
    conv_1d_invoker(n1, M, plan);
    break;
  case 2:
    conv_2d_invoker(nf1, nf2, M, plan);
    break;
  case 3:
    conv_3d_invoker(nf1, nf2, nf3, M, plan);
  default:
    ier = 1; // error
    break;
  }

  return ier;
}

int curaff_partial_conv(){
  
  // improved WS
  // invoke the partial 3d conv, calcualte the conv result and saved the result to plan->fw
  // directly invoke, not packed into function


  // WS
  return 0;
}