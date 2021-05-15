//------ convolutional gridding -------
/*
    1. W-term gridding
    2. u v gridding
*/

#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <helper_cuda.h>
//#include <thrust/extrema.h>
#include "conv.h"

static __inline__ __device__ int kerval(PCS x, PCS es_c, PCS es_beta){
	//not using the fast kernel evaluation
	return exp(es_beta * (sqrt(1.0 - es_c*x*x)));
}

static __inline__ __device__
void val_kernel_vec(PCS *ker, const PCS x, const double w, const double es_c, 
					 const double es_beta)
{
	//get vector of kernel function values
	for(int i=0; i<w; i++){
		ker[i] = kerval(abs(x+i), es_c, es_beta);		
	}
}

// __global__ void print_res(CUCPX *fw){
// 	int idx = threadIdx.x + blockDim.x * blockIdx.x;
// 	printf("the value of %d is %2.2g\n",idx,fw[idx].x);
// }

// 2D for w-stacking. 1D + 2D for improved WS will consume more memory
__global__ void conv_2d_nputsdriven(PCS *x, PCS *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, PCS es_c, PCS es_beta, int pirange, INT_M* cell_loc)
{
	/*
		x, y - range [-pi,pi)
		c - complex number
		fw - result
		M - number of nupts
		ns - kernel width
		nf1, nf2 - upts
		es_ - gridding kernel related factors
		pirange - 1
		cell_loc - location of nupts in grid cells
	*/
	//need to revise 
	int xstart,ystart,xend,yend;
	int ix, iy;
	int outidx;
	PCS ker1[MAX_KERNEL_WIDTH];
	PCS ker2[MAX_KERNEL_WIDTH];

	PCS temp1, temp2;
	int idx;
	//__shared__ CUCPX s_c[blockDim.x];
	//assert(pirange==1);// check

	for(idx = blockIdx.x * blockDim.x + threadIdx.x;idx<M;idx+=gridDim.x*blockDim.x){
		
		//value of x and w, rescale to [0,N) and get the locations
		temp1 = ((x[idx]<0?(x[idx]+PI):x[idx]) * M_1_2PI * nf1); 
		temp2 = ((y[idx]<0?(y[idx]+PI):y[idx]) * M_1_2PI * nf2);
		if(cell_loc!=NULL){
			cell_loc[idx].x = (int)(temp1);	//need to save?
			cell_loc[idx].y = (int)(temp2); //change to int2
		}
		//s_c[idx] = c[idx];
		//change rescaled to cell_loc
		xstart = ceil(temp1 - ns/2.0);
		ystart = ceil(temp2 - ns/2.0);
		xend = floor(temp1 + ns/2.0);
		yend = floor(temp2 + ns/2.0);
		
		PCS x_1=(PCS)xstart-temp1; //cell
		PCS y_1=(PCS)ystart-temp2;
		val_kernel_vec(ker1,x_1,ns,es_c,es_beta);
		val_kernel_vec(ker2,y_1,ns,es_c,es_beta);
		for(int yy=ystart; yy<=yend; yy++){
			temp1=ker2[yy-ystart];
			for(int xx=xstart; xx<=xend; xx++){
				ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
				iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				outidx = ix+iy*nf1;
				temp2=ker1[xx-xstart];
				PCS kervalue=temp1*temp2;
				atomicAdd(&fw[outidx].x, c[idx].x*kervalue);
				atomicAdd(&fw[outidx].y, c[idx].y*kervalue);
			}
		}
		//if((idx/blockDim.x+1)*blockDim.x<M){__syncthreads();}
	}

}

__global__
void conv_3d_nputsdriven(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, PCS es_c, PCS es_beta, int pirange, INT_M* cell_loc)
{
	/*
		x, y, z - range [-pi,pi)
		c - complex number
		fw - result
		M - number of nupts
		ns - kernel width
		nf1, nf2, nf3 - upts
		es_ - gridding kernel related factors
		pirange - 1
		cell_loc - location of nupts in grid cells
	*/
	
	int idx;
	idx = blockDim.x*blockIdx.x+threadIdx.x;
	int xx, yy, zz, ix, iy, iz;
	int outidx;
	
	PCS ker1[MAX_KERNEL_WIDTH];
	PCS ker2[MAX_KERNEL_WIDTH];
	PCS ker3[MAX_KERNEL_WIDTH];
	
	PCS temp1, temp2, temp3;
	
	assert(pirange==1);// check, the x y z should be in range [-pi,pi)
	
	for(idx=blockDim.x*blockIdx.x+threadIdx.x; idx<M; idx+=blockDim.x*gridDim.x){
		
		//value of x and w, rescale to [0,N) and get the locations
		// if pirange = 2 need to change
		temp1 = ((x[idx]<0?(x[idx]+PI):x[idx]) * M_1_2PI * nf1); 
		temp2 = ((y[idx]<0?(y[idx]+PI):y[idx]) * M_1_2PI * nf2);
		temp3 = ((z[idx]<0?(z[idx]+PI):z[idx]) * M_1_2PI * nf3);
		
		// add if cell = NULL skip
		if(cell_loc!=NULL){
			cell_loc[idx].x = (int)(temp1);	//need to save?
			cell_loc[idx].y = (int)(temp2); 
			cell_loc[idx].z = (int)(temp3);
		}
		//change rescaled to cell_loc
		

		int xstart = ceil(temp1 - ns/2.0);
		int ystart = ceil(temp2 - ns/2.0);
		int zstart = ceil(temp3 - ns/2.0);
		int xend = floor(temp1 + ns/2.0);
		int yend = floor(temp2 + ns/2.0);
		int zend = floor(temp3 + ns/2.0);

		PCS x1=(PCS)xstart-temp1;
		PCS y1=(PCS)ystart-temp2;
		PCS z1=(PCS)zstart-temp3;
		


		val_kernel_vec(ker1,x1,ns,es_c,es_beta);
		val_kernel_vec(ker2,y1,ns,es_c,es_beta);
		val_kernel_vec(ker3,z1,ns,es_c,es_beta);

		for(zz=zstart; zz<=zend; zz++){
			temp3=ker3[zz-zstart];
			for(yy=ystart; yy<=yend; yy++){
				temp2=ker2[yy-ystart];
				for(xx=xstart; xx<=xend; xx++){
					//due to the peroid, the index out of range need to be handle
					ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
					iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
					iz = zz < 0 ? zz+nf3 : (zz>nf3-1 ? zz-nf3 : zz);
					outidx = ix+iy*nf1+iz*nf1*nf2;

					temp1=ker1[xx-xstart];
					PCS kervalue=temp1*temp2*temp3;
					atomicAdd(&fw[outidx].x, c[idx].x*kervalue);
					atomicAdd(&fw[outidx].y, c[idx].y*kervalue);
					//printf("the out id %d kervalue %2.2g\n",outidx,kervalue);
				}
			}
		}
		//if((idx/blockDim.x+1)*blockDim.x<M){ __syncthreads(); }
	}
	
}



/*
PCS evaluate_kernel(PCS x, const conv_opts &opts)
/* ES ("exp sqrt") kernel evaluation at single real argument:
      phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg common/onedim_* 2/17/17  //?
{
  if (abs(x)>=opts.ES_halfwidth)
    // if spreading/FT careful, shouldn't need this if, but causes no speed hit
    return 0.0;
  else
    return exp(opts.ES_beta * sqrt(1.0 - opts.ES_c*x*x));
}
*/

