#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <thrust/complex.h>
#include <algorithm>
//#include <thrust>
using namespace thrust;


#include "ragridder_plan.h"
#include "conv_invoker.h"
#include "cuft.h"
#include "deconv.h"
#include "cugridder.h"
#include "precomp.h"
#include "utils.h"


int main(int argc, char *argv[])
{
	/* Input: M, N1, N2, epsilon method
		method - conv method
		M - number of randomly distributed points
		N1, N2 - output size
		epsilon - tolerance
	*/
	int ier = 0;
	int N = 16;
	PCS sigma = 2.0; // upsampling factor
	int M = 30;

	
	PCS epsilon = 1e-6;
	
	int kerevalmeth = 0;
	
	int method=0;

	//gpu_method == 0, nupts driven

	//int ier;
	PCS *u;
	CPX *c;
	u = (PCS *)malloc(M  * sizeof(PCS)); //Allocates page-locked memory on the host.
	c = (CPX *)malloc(M  * sizeof(CPX));
	PCS *d_u;
	CUCPX *d_c, *d_fk;
	CUCPX *d_fw;
	checkCudaErrors(cudaMalloc(&d_u, M  * sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_c, M  * sizeof(CUCPX)));
    /// pixel size 
	// generating data
	for (int i = 0; i < M; i++)
	{
		u[i] = randm11()*PI; //xxxxx
		c[i].real(randm11()); // M vis per channel, weight?
		c[i].imag(randm11());
		// wgt[i] = 1;
	}

	PCS *k = (PCS*) malloc(sizeof(PCS)*N*10);
	// PCS pixelsize = 0.01;
	for (size_t i = 0; i < N; i++)
	{
		/* code */
		k[i] = (int)i-N/2;
		// k[i] = randm11();
	}
	
	// double a[5] = {-PI/2, -PI/3, 0, PI/3, PI/2}; // change to random data
	// for(int i=0; i<M; i++){
	// 	u[i] = a[i/5];
	// 	v[i] = a[i%5];
	// }
#ifdef DEBUG
	printf("origial input data...\n");
	for(int i=0; i<M; i++){
		printf("%.3lf ",u[i]);
	}
	printf("\n");
	for(int i=0; i<M; i++){
		printf("%.3lf ",c[i].real());
	}
	printf("\n");
#endif
	// ignore the tdirty
	// how to convert ms to vis

	//printf("generated data, x[1] %2.2g, y[1] %2.2g , z[1] %2.2g, c[1] %2.2g\n",x[1] , y[1], z[1], c[1].real());

	// Timing begin
	//data transfer
	checkCudaErrors(cudaMemcpy(d_u, u, M * sizeof(PCS), cudaMemcpyHostToDevice)); //u
	checkCudaErrors(cudaMemcpy(d_c, c, M * sizeof(CUCPX), cudaMemcpyHostToDevice));

	/* ----------Step2: plan setting------------*/
	curafft_plan *plan;

	plan = new curafft_plan();
    memset(plan, 0, sizeof(curafft_plan));

	int direction = 1; //inverse
	
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

    ier = setup_conv_opts(plan->copts, epsilon, sigma, 1, direction, kerevalmeth); //check the arguements

	if(ier!=0)printf("setup_error\n");

    // plan setting
    // cuda stream malloc in setup_plan
    

    int nf1 = get_num_cells(M,plan->copts);

    
    plan->dim = 1;
    setup_plan(nf1, 1, 1, M, d_u, NULL, NULL, d_c, plan);

	plan->ms = N; ///!!!
	plan->mt = 1;
	plan->mu = 1;
    plan->execute_flow = 1;
	int iflag = direction;
    int fftsign = (iflag>=0) ? 1 : -1;

	plan->iflag = fftsign; //may be useless| conflict with direction
	plan->batchsize = 1;

    plan->copts.direction = direction; // 1 inverse, 0 forward
    PCS *d_fwkerhalf;
    checkCudaErrors(cudaMalloc((void**)&d_fwkerhalf,sizeof(PCS)*(N)));
	cudaMemset(d_fwkerhalf,0,sizeof(PCS)*N);
    PCS *d_k;
    checkCudaErrors(cudaMalloc((void**)&d_k,sizeof(PCS)*(N)));
    checkCudaErrors(cudaMemcpy(d_k,k,sizeof(PCS)*(N),cudaMemcpyHostToDevice));
    fourier_series_appro_invoker(d_fwkerhalf,d_k,plan->copts, N,nf1/2+1); // correction with k, may be wrong, k will be free in this function
	
	
	// printf("begining...\n");
	// fourier_series_appro_invoker(d_fwkerhalf,plan->copts,nf1/2+1);
	PCS *fwkerhalf = (PCS *)malloc(sizeof(PCS)*(N));
	cudaMemcpy(fwkerhalf, d_fwkerhalf, sizeof(PCS)*(N), cudaMemcpyDeviceToHost);
	printf("correction factor printing method1...\n");
	for (size_t i = 0; i < N; i++)
	{
		/* code */
		printf("%lf ",fwkerhalf[i]);
	}
	printf("\n");
	fourier_series_appro_invoker(d_fwkerhalf,plan->copts,nf1/2+1);
	cudaMemcpy(fwkerhalf, d_fwkerhalf, sizeof(PCS)*(nf1/2+1), cudaMemcpyDeviceToHost);
	printf("correction factor printing method2...\n");
	for (size_t i = 0; i < N/2+1; i++)
	{
		/* code */
		printf("%lf ",fwkerhalf[i]);
	}
	printf("\n");
	// for(int i=0; i<nf1/2+1; i++){
	// 	k[i] = i;
	// }
	// PCS *d_temp;
	// checkCudaErrors(cudaMalloc((void**)&d_temp,sizeof(PCS)*(nf1/2+1)));
	// checkCudaErrors(cudaMemcpy(d_temp,k,sizeof(PCS)*(nf1/2+1),cudaMemcpyHostToDevice));
	// fourier_series_appro_invoker(d_fwkerhalf,d_temp,plan->copts,nf1/2+1,nf1/2+1);
	// cudaMemcpy(fwkerhalf, d_fwkerhalf, sizeof(PCS)*(N), cudaMemcpyDeviceToHost);
	// printf("correction factor printing method2...\n");
	// for (size_t i = 0; i < N; i++)
	// {
	// 	/* code */
	// 	printf("%lf ",fwkerhalf[i]);
	// }
	// printf("\n");

    
	// fw (conv res set)
	checkCudaErrors(cudaMalloc((void**)&d_fw,sizeof(CUCPX)*nf1));
	checkCudaErrors(cudaMemset(d_fw, 0, sizeof(CUCPX)*nf1));
	plan->fw = d_fw;
	// fk malloc and set
	checkCudaErrors(cudaMalloc((void**)&d_fk,sizeof(CUCPX)*N));
	plan->fk = d_fk;

	// calulating result
	curafft_conv(plan);
	CPX *fw = (CPX *)malloc(sizeof(CPX)*nf1);
	cudaMemcpy(fw,plan->fw,sizeof(CUCPX)*nf1,cudaMemcpyDeviceToHost);
#ifdef DEBUG
	printf("conv result printing...\n");
	
	PCS temp_res=0;
	
	for (size_t i = 0; i < nf1; i++)
	{
		/* code */
		printf("%lf ",fw[i].real());
	}
	printf("\n");
	
#endif
	CPX *fk = (CPX *)malloc(sizeof(CPX)*N);
	memset(fk,0,sizeof(CPX)*N);
	// dft
	for (size_t i = 0; i < N; i++)
	{
		/* code */
		for (size_t j = 0; j < nf1; j++)
		{

			fk[i] += fw[j<nf1/2?j+nf1/2:j-nf1/2]*exp(k[i]*(j)/(PCS)nf1*2*PI*IMA);
		}
		
	}
	
printf("dft result printing...\n");
for (size_t i = 0; i < N; i++)
{
	/* code */
	printf("%lf ",fk[i].real());
}
printf("\n");

	 cufftHandle fftplan;
    int n[] = { plan->nf1};
    int inembed[] = { plan->nf1};
	int onembed[] = { plan->nf1};
    
	// cufftCreate(&fftplan);
	// cufftPlan2d(&fftplan,n[0],n[1],CUFFT_TYPE);
    // the bach size sets as the num of w when memory is sufficent. Alternative way, set as a smaller number when memory is insufficient.
    // and handle this piece by piece 
    
	cufftPlanMany(&fftplan,1,n,inembed,1,inembed[0],
		onembed,1,onembed[0],CUFFT_TYPE,plan->nf3); //need to check and revise (the partial conv will be differnt)
    plan->fftplan = fftplan; 

	// printf("correction factor printing...\n");
	// for(int i=0; i<N1/2; i++){
	// 	printf("%.3g ",fwkerhalf1[i]);
	// }
	// printf("\n");
	// for(int i=0; i<N2/2; i++){
	// 	printf("%.3g ",fwkerhalf2[i]);
	// }
	// printf("\n");
	// deconv
	//PCS *fwkerhalf = (PCS *)malloc(sizeof(PCS)*(N));
	cudaMemcpy(fwkerhalf, d_fwkerhalf, sizeof(PCS)*(N), cudaMemcpyDeviceToHost);

	for(int i=0; i<N; i++){
		fk[i] = fk[i] / fwkerhalf[abs(i-N/2)];
	}

	
	// result printing
	printf("final result printing...\n");
	for(int i=0; i<N; i++){
		printf("%.10lf ",fk[i].real());
		
	}
	printf("\n");
	printf("ground truth printing...\n");
	for (size_t i = 0; i < N; i++)
	{
		/* code */
		fk[i] = 0;
		for(int j=0; j<M; j++){
			fk[i] += c[j]*exp(k[i]*u[j]*IMA);
		}
	}
	
	for(int i=0; i<N; i++){
		printf("%.10lf ",fk[i].real());
		
	}
	printf("\n");
	
	//free
	curafft_free(plan);
	free(fk);
	free(u);
	free(c);

	return ier;
}