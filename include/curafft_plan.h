#ifndef __CURAFFT_PLAN_H__
#define __CURAFFT_PLAN_H__


#include <cstdlib>
#include <cufft.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "dataType.h"
#include "curafft_opts.h"
#include "utils.h"
#include "visibility.h"

#define MAX_KERNEL_WIDTH 16

struct conv_opts{
  /*
    options for convolutional gridding process
    kw - w, the kernel width (number of grid cells)
    direction - 1 means inverse NU->U, 0 means forward interpolate U->NU //changed
    pirange - 0: coords in [0,N), 1 coords in [-pi,pi), 2 coords in [-lamda/2,lamda) * resolution / fov for scaling
    upsampfac - sigma, upsampling factor, default 2.0
    ES_beta
    ES_halfwidth
    ES_c
  */
  int kw;   //kernel width // also need to take factors in improved ws into consideration
  int direction;
  PCS pirange; // [-pi,pi)
  PCS upsampfac;
  // ES kernel specific...
  PCS ES_beta;
  PCS ES_halfwidth;
  PCS ES_c;//default 4/kw^2 for reusing
};

struct curafft_plan
{
    curafft_opts opts;
    conv_opts copts;
	//cufft
    cufftHandle fftplan;
	//A stream in CUDA is a sequence of operations that execute on the device in the order in which they are issued 
	//by the host code. While operations within a stream are guaranteed to execute in the prescribed order, operations
	//in different streams can be interleaved and, when possible, they can even run concurrently.
	cudaStream_t *streams;

    //int type;

	//suppose the N_u = N_l
	PCS *d_u;
	PCS *d_v;
	PCS *d_w;
	PCS *d_c;
	int dim; //dimension support for 1,2,3D
	int mode_flag; // FFTW (0) style or CMCL-compatible mode ordering (1)
	int M; //NU
	int nf1; // UPTS after upsampling
	int nf2;
	int nf3; //number of w after gridding
	int ms; // number of Fourier modes N1
	int mt; // N2
	int mu; // N3
	int ntransf;
	int iflag;
	int batchsize;
	int execute_flow;//may be useless

	//int totalnumsubprob;
	int byte_now; //always be set to be 0
	PCS *fwkerhalf1; //used for not just spread only
	PCS *fwkerhalf2;
	PCS *fwkerhalf3;

	CUCPX *fw; // conv res
	CUCPX *fk; // fft res

	int *idxnupts;//length: #nupts, index of the nupts in the bin-sorted order (size is M) abs location in bin
	int *sortidx; //length: #nupts, order inside the bin the nupt belongs to (size is M) local position in bin
	INT_M *cell_loc; // length: #nupts, location in grid cells for 2D case


	//----for GM-sort method----
	int *binsize; //length: #bins, number of nonuniform ponits in each bin //one bin can contain add to gpu_binsizex*gpu_binsizey points
	int *binstartpts; //length: #bins, exclusive scan of array binsize // binsize after scan
	
    /*


	// Arrays that used in subprob method
	int *numsubprob; //length: #bins,  number of subproblems in each bin
	
	int *subprob_to_bin;//length: #subproblems, the bin the subproblem works on 
	int *subprobstartpts;//length: #bins, exclusive scan of array numsubprob
    

	// Extra arrays for Paul's method
	int *finegridsize;
	int *fgstartpts;
    
	// Arrays for 3d (need to sort out)
	int *numnupts;
	int *subprob_to_nupts;
    */

};

#endif
