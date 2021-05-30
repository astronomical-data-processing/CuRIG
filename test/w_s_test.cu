#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <thrust/complex.h>
#include <algorithm>
//#include <thrust>
using namespace thrust;
using namespace std::complex_literals;

#include "conv_invoker.h"
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
	CPX *ms, *fk;
	CPX *vis;
	u = (PCS *)malloc(nrow * sizeof(PCS)); //Allocates page-locked memory on the host.
	v = (PCS *)malloc(nrow * sizeof(PCS));
	w = (PCS *)malloc(nrow * sizeof(PCS));
	ms = (CPX *)malloc(nrow * sizeof(CPX));
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
		u[i] = M_PI * randm11() / 0.5 / (deg_per_pixelx * f0 / SPEEDOFLIGHT); //will change for different freq?
		v[i] = M_PI * randm11() / 0.5 / (deg_per_pixelx * f0 / SPEEDOFLIGHT);
		w[i] = M_PI * randm11() / 0.5 / (deg_per_pixelx * f0 / SPEEDOFLIGHT);
		ms[i].real(randm11() / 0.5);
		ms[i].imag(randm11() / 0.5);
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

	// getWgtIndices ???
	/* ----------Step2: Setup conv opts---------------
        add deg_per_pix = fov / resolution and correction related variables
    */

	curafft_plan *h_plan = new curafft_plan();
	memset(h_plan, 0, sizeof(curafft_plan));

	// opts and copts setting
	h_plan->opts.gpu_conv_only = 1;
	h_plan->opts.gpu_gridder_method = method;
	h_plan->opts.gpu_kerevalmeth = kerevalmeth;
	h_plan->opts.gpu_sort = 1;
	// h_plan->copts.pirange = 1;
	// some plan setting
	h_plan->w_term_method = w_term_method;

	int ier = setup_conv_opts(h_plan->copts, tol, sigma, kerevalmeth); //check the arguements

	if (ier != 0)
		printf("setup_error\n");

	// plan setting

	ier = setup_plan(N1, N2, M, d_x, d_y, d_z, d_c, h_plan); //cautious the number of plane using N1 N2 to get nf1 nf2

	//printf("the num of w %d\n",h_plan->num_w);
	int nf1 = h_plan->nf1;
	int nf2 = h_plan->nf2;
	h_plan->num_w = 2;
	int nf3 = h_plan->num_w; //correctness checking
	printf("the kw is %d\n", h_plan->copts.kw);
	int f_size = N1 * N2;
	//need to revise
	fk = (CPX *)malloc(sizeof(CPX) * f_size);
	checkCudaErrors(cudaMalloc(&d_fk, f_size * sizeof(CUCPX)));

	h_plan->fk = d_fk;

	std::cout << std::scientific << std::setprecision(3); //setprecision not define

	cudaEvent_t cuda_start, cuda_end;

	float kernel_time;

	cudaEventCreate(&cuda_start);
	cudaEventCreate(&cuda_end);

	cudaEventRecord(cuda_start);

	// convolution
	curafft_conv(h_plan); //add to include
	cudaEventRecord(cuda_end);

	curafft_free(h_plan);
	cudaEventSynchronize(cuda_start);
	cudaEventSynchronize(cuda_end);

	cudaEventElapsedTime(&kernel_time, cuda_start, cuda_end);

	checkCudaErrors(cudaDeviceSynchronize());
	// need to revise
	checkCudaErrors(cudaMemcpy(fk, d_fk, sizeof(CUCPX) * f_size, cudaMemcpyDeviceToHost));

	//int nf3 = h_plan->num_w;
	printf("Method %d (nupt driven) %ld NU pts to #%d U pts in %.3g s\n",
		   h_plan->opts.gpu_gridder_method, M, nf1 * nf2 * nf3, kernel_time / 1000);

	/*
	std::cout<<"[result-input]"<<std::endl;
	for(int k=0; k<nf3; k++){
		for(int j=0; j<nf2; j++){
			for (int i=0; i<nf1; i++){
				printf(" (%2.3g,%2.3g)",fw[i+j*nf1+k*nf2*nf1].real(),
					fw[i+j*nf1+k*nf2*nf1].imag() );
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}
	std::cout<<"----------------------------------------------------------------"<<std::endl;
	*/
	checkCudaErrors(cudaFree(d_x));
	checkCudaErrors(cudaFree(d_y));
	checkCudaErrors(cudaFree(d_z));
	checkCudaErrors(cudaFree(d_c));
	checkCudaErrors(cudaFree(d_fk));

	checkCudaErrors(cudaDeviceReset());
	free(x);
	free(y);
	free(z);
	free(c);
	free(fk);

	return 0;
}