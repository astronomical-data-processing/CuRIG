#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <thrust/complex.h>
#include <algorithm>
//#include <thrust>
using namespace thrust;
using namespace std::complex_literals;

#include "ragridder_plan.h"
#include "conv_invoker.h"
#include "cugridder.h"
#include "utils.h"

///conv improved WS, method 0 correctness cheak

int main(int argc, char *argv[])
{
	// suppose there is just one channel
	// range of uvw [-lamda/2,lamda/2]， rescale with factor resolution / fov compatible with l
	// l and m need to be converted into pixels
	/* Input: nrow, nchan, nxdirty, nydirty, fov, epsilon
		row - number of visibility
		nchan - number of channels
		nxdirty, nydirty - image size
		fov - field of view
		epsilon - tolerance
	*/
	int ier = 0;
	if (argc < 5)
	{
		fprintf(stderr,
				"Usage: spread3d method nupts_distr nf1 nf2 nf3 [maxsubprobsize [M [tol [kerevalmeth [sort]]]]]\n"
				"Arguments:\n"
				"  method: One of\n"
				"    0: nupts driven,\n"
				"    2: sub-problem, or\n"
				"    4: block gather (each nf must be multiple of 8).\n"
				"  w_term_method: \n"
				"    0: w-stacking\n"
				"    1: improved w-stacking\n"
				"  nxdirty, nydirty : image size.\n"
				"  nrow: The number of non-uniform points.\n"
				"  fov: Field of view.\n"
				"  nchan: number of chanels (default 1)"
				"  epsilon: NUFFT tolerance (default 1e-6).\n"
				"  kerevalmeth: Kernel evaluation method; one of\n"
				"     0: Exponential of square root (default), or\n"
				"     1: Horner evaluation.\n");
		return 1;
	}

	int nxdirty, nydirty;
	PCS sigma = 2.0; // upsampling factor
	int nrow, nchan;
	float fov;

	double inp;
	int method;
	sscanf(argv[1], "%d", &method);
	int w_term_method;
	sscanf(argv[2], "%d", &w_term_method);
	sscanf(argv[3], "%lf", &inp);
	nxdirty = (int)inp;
	sscanf(argv[4], "%lf", &inp);
	nydirty = (int)inp;
	sscanf(argv[5], "%lf", &inp);
	nrow = (int)inp;
	sscanf(argv[6], "%lf", &inp);
	fov = inp;

	nchan = 1;
	if (argc > 7)
	{
		sscanf(argv[7], "%lf", &inp);
		nchan = (int)inp; // so can read 1e6 right!
	}

	PCS epsilon = 1e-6;
	if (argc > 8)
	{
		sscanf(argv[8], "%lf", &inp);
		epsilon = (PCS)inp; // so can read 1e6 right!
	}

	int kerevalmeth = 0;
	if (argc > 9)
	{
		sscanf(argv[9], "%d", &kerevalmeth);
	}

	// degree per pixel (unit radius)
	PCS deg_per_pixelx = fov / 180.0 * PI / (PCS)nxdirty;
	PCS deg_per_pixely = fov / 180.0 * PI / (PCS)nydirty;
	// chanel setting
	PCS f0 = 1e9;
	PCS freq = (PCS *)malloc(sizeof(PCS) * nchan);
	for (int i = 0; i < nchan; i++)
	{
		freq[i] = f0 + i / (double)nchan * fov; //!
	}

	//improved WS stacking 1,
	//gpu_method == 0, nupts driven

	//N1 = 5; N2 = 5; M = 25; //for correctness checking
	//int ier;
	PCS *u, *v, *w;
	CUCPX *fk;
	CPX *vis;
	PCS *wgt; //currently no mask
	u = (PCS *)malloc(nrow * sizeof(PCS)); //Allocates page-locked memory on the host.
	v = (PCS *)malloc(nrow * sizeof(PCS));
	w = (PCS *)malloc(nrow * sizeof(PCS));
	vis = (CPX *)malloc(nrow * sizeof(CPX));
	PCS *d_u, *d_v, *d_w;
	CUCPX *d_vis, *d_fk;
	checkCudaErrors(cudaMalloc(&d_x, nrow * sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_y, nrow * sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_z, nrow * sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_vis, nrow * sizeof(CUCPX)));

	// generating data
	for (int i = 0; i < nrow; i++)
	{
		u[i] = randm11() / 0.5 / (deg_per_pixelx * f0 / SPEEDOFLIGHT); //will change for different freq?
		v[i] = randm11() / 0.5 / (deg_per_pixelx * f0 / SPEEDOFLIGHT);
		w[i] = randm11() / 0.5 / (deg_per_pixelx * f0 / SPEEDOFLIGHT);
		vis[i].real(randm11() / 0.5);
		vis[i].imag(randm11() / 0.5);
		wgt[i] = 1;
	}
	// ignore the tdirty
	// how to convert ms to vis

	//printf("generated data, x[1] %2.2g, y[1] %2.2g , z[1] %2.2g, c[1] %2.2g\n",x[1] , y[1], z[1], c[1].real());

	// Timing begin
	//data transfer
	checkCudaErrors(cudaMemcpy(d_u, u, nrow * sizeof(PCS), cudaMemcpyHostToDevice)); //u
	checkCudaErrors(cudaMemcpy(d_v, v, nrow * sizeof(PCS), cudaMemcpyHostToDevice)); //v
	checkCudaErrors(cudaMemcpy(d_w, w, nrow * sizeof(PCS), cudaMemcpyHostToDevice)); //w
	checkCudaErrors(cudaMemcpy(d_vis, vis, nrow * sizeof(CUCPX), cudaMemcpyHostToDevice));

	

	/* -----------Step1: Baseline setting--------------
	skip negative v
    uvw, nrow = M, shift, mask, f_over_c (fixed due to single channel)
    */
	int shift = 0;
	while ((int(1) << shift) < nchan)
		++shift;
	int mask = (int(1) << shift) - 1; // ???
	PCS f_over_c = (PCS*) malloc(sizeof(PCS)*nchan);
	for(int i=0; i<nchan; i++){
		f_over_c[i] = freq[i]/SPEEDOFLIGHT;
	}

	/* ----------Step2: cugridder------------*/
	// plan setting
	curafft_plan *plan;

	ragridder_plan *gridder_plan;
	
	gridder_plan->kv.u = u;
	gridder_plan->kv.v = v;
	gridder_plan->kv.w = w;
	gridder_plan->kv.frequency = freq;
	gridder_plan->kv.vis = vis;
	gridder_plan->kv.weight = wgt;

	int direction = 1; //inverse

	ier = gridder_setting(nxdirty,nydirty,method,kerevalmeth,w_term_method,direction,sigma,0,1,nrow,fov,d_u,d_v,d_w,d_vis
		,plan,gridder_plan);
	if(ier == 1){
		printf("errors in gridder setting\n");
		return ier;
	}
	// fk(image) malloc and set
	checkCudaErrors(cudaMalloc((void**)&fk,sizeof(CUCPX)*nydirty*nxdirty));
	plan->fk = fk;

	gridder_plan->dirty_image = (CPX *)malloc(sizeof(CPX)*nxdirty*nydirty*nchan);
	
	// how to use weight flag and frequency
	for(int i=0; i<nchan; i++){
		ier = gridder_exectuion(plan);
		if(ier == 1){
			printf("errors in gridder execution\n");
			return ier;
		}
		checkCudaErrors(cudaMemcpy(gridder_plan->dirty_image+i*nxdirty*nydirty, fk, sizeof(CUCPX)*nydirty*nxdirty,
			cudaMemcpyDeviceToHost));
	}

	ier = gridder_destroy(plan, gridder_plan);
	if(ier == 1){
		printf("errors in gridder destroy\n");
		return ier;
	}
	return 0;
}