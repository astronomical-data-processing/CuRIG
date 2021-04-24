#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <thrust/complex.h>
#include <algorithm>
//#include <thrust>
using namespace thrust;
using namespace std::complex_literals;
#include "../src/FT/conv_invoker.h"
//#include "../include/curafft.h"
#include "../include/curafft_opts.h"
#include "../src/utils.h"


int main(int argc, char* argv[]){
	//improved WS stacking 1,
	//gpu_method == 0, nupts driven
    int nf1, nf2;
	PCS sigma = 2.0;
	int M;
	if (argc<6) {
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
			"  nf1, nf2 : image size.\n"
			"  M: The number of non-uniform points.\n"
			"  tol: NUFFT tolerance (default 1e-6).\n"
			"  kerevalmeth: Kernel evaluation method; one of\n"
			"     0: Exponential of square root (default), or\n"
			"     1: Horner evaluation.\n"
		);
		return 1;
	}
	double w;
	int method;
	sscanf(argv[1],"%d",&method);
	int w_term_method;
	sscanf(argv[2],"%d",&w_term_method);
	sscanf(argv[3],"%lf",&w); nf1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[4],"%lf",&w); nf2 = (int)w;  // so can read 1e6 right!
	sscanf(argv[5],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
	

	PCS tol=1e-6;
	if(argc>6){
		sscanf(argv[6],"%lf",&w); tol  = (PCS)w;  // so can read 1e6 right!
	}

	int kerevalmeth=0;
	if(argc>7){
		sscanf(argv[7],"%d",&kerevalmeth);
	}

    // fov and 1 pixel corresonding to pix_deg degree

	//int ier;
	PCS *x, *y, *z;
	CPX *c, *fw;
	cudaMallocHost(&x, M*sizeof(PCS)); //Allocates page-locked memory on the host.
	cudaMallocHost(&y, M*sizeof(PCS));
	cudaMallocHost(&z, M*sizeof(PCS));
	cudaMallocHost(&c, M*sizeof(CPX));
	//cudaMallocHost(&fw,nf1*nf2*nf3*sizeof(CPX)); //malloc after plan setting

	PCS *d_x, *d_y, *d_z;
	CUCPX *d_c, *d_fw;
	checkCudaErrors(cudaMalloc(&d_x,M*sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_y,M*sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_z,M*sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_c,M*sizeof(CUCPX)));
	//checkCudaErrorsCudaErrors(cudaMalloc(&d_fw,nf1*nf2*nf3*sizeof(CUCPX)));

    //generating data
    int nupts_distribute = 0;
	switch(nupts_distribute){
		case 0: //uniform
			{
				for (int i = 0; i < M; i++) {
					x[i] = M_PI*randm11();
					y[i] = M_PI*randm11();
					z[i] = M_PI*randm11();
					c[i].real(randm11());
					c[i].imag(randm11());
				}
			}
			break;
		case 1: // concentrate on a small region
			{
				for (int i = 0; i < M; i++) {
					x[i] = M_PI*rand01()/nf1*16;
					y[i] = M_PI*rand01()/nf2*16;
					z[i] = M_PI*rand01()/nf2*16;
					c[i].real(randm11());
					c[i].imag(randm11());
				}
			}
			break;
		default:
			std::cerr << "not valid nupts distr" << std::endl;
			return 1;
	}

    //data transfer
	checkCudaErrors(cudaMemcpy(d_x,x,M*sizeof(PCS),cudaMemcpyHostToDevice)); //u
	checkCudaErrors(cudaMemcpy(d_y,y,M*sizeof(PCS),cudaMemcpyHostToDevice)); //v
	checkCudaErrors(cudaMemcpy(d_z,z,M*sizeof(PCS),cudaMemcpyHostToDevice)); //w
	checkCudaErrors(cudaMemcpy(d_c,c,M*sizeof(CUCPX),cudaMemcpyHostToDevice));

    curafft_plan *h_plan = new curafft_plan();
    memset(h_plan, 0, sizeof(curafft_plan));
	
    // opts and copts setting
    h_plan->opts.gpu_conv_only = 1;
    h_plan->opts.gpu_method = method;
	h_plan->opts.gpu_kerevalmeth = kerevalmeth;

    setup_conv_opts(h_plan->copts,sigma, tol, kerevalmeth); //check the arguements

    // w term related setting
    //setup_grid_wsize();
    
    // plan setting
	
    setup_plan(nf1, nf2, M, d_x, d_y, d_z, d_c, h_plan); //add to .h file


    cudaMallocHost(&fw,nf1*nf2*h_plan->num_w*sizeof(CPX)); //malloc after plan setting
    checkCudaErrors(cudaMalloc(&d_fw,nf1*nf2*h_plan->num_w*sizeof(CUCPX)));

	//binsize, obinsize need to be set here, since SETUP_BINSIZE() is not 
	//called in spread, interp only wrappers.
    /*
	if(dplan->opts.gpu_method == 4)
	{
		dplan->opts.gpu_binsizex=4;
		dplan->opts.gpu_binsizey=4;
		dplan->opts.gpu_binsizez=4;
		dplan->opts.gpu_obinsizex=8;
		dplan->opts.gpu_obinsizey=8;
		dplan->opts.gpu_obinsizez=8;
		dplan->opts.gpu_maxsubprobsize=maxsubprobsize;
	}
	if(dplan->opts.gpu_method == 2)
	{
		dplan->opts.gpu_binsizex=16;
		dplan->opts.gpu_binsizey=16;
		dplan->opts.gpu_binsizez=2;
		dplan->opts.gpu_maxsubprobsize=maxsubprobsize;
	}
	if(dplan->opts.gpu_method == 1)
	{
		dplan->opts.gpu_binsizex=16;
		dplan->opts.gpu_binsizey=16;
		dplan->opts.gpu_binsizez=2;
	}
    */

	std::cout<<std::scientific<<std::setprecision(3);//setprecision not define


	//CNTime timer; //
	/*warm up gpu
	char *a;
	timer.restart();
	checkCudaErrors(cudaMalloc(&a,1));
	// cout<<"[time  ]"<< " (warm up) First cudamalloc call " << timer.elapsedsec()
	//	<<" s"<<endl<<endl;



	timer.restart();
	*/

	cudaEvent_t cuda_start, cuda_end;

    float kernel_time;
    
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_end);

    cudaEventRecord(cuda_start);

    // convolution
    curafft_conv(h_plan); //add to include
	cudaEventRecord(cuda_end);

    cudaEventSynchronize(cuda_start);
    cudaEventSynchronize(cuda_end);

    cudaEventElapsedTime(&kernel_time, cuda_start, cuda_end);


    checkCudaErrors(cudaDeviceSynchronize());
	
	int nf3 = h_plan->num_w;
	printf("[Method %d] %ld NU pts to #%d U pts in %.3g s\n",
			h_plan->opts.gpu_method,M,nf1*nf2*nf3,kernel_time);
	
	

	std::cout<<"[result-input]"<<std::endl;
	for(int k=0; k<nf3; k++){
		for(int j=0; j<nf2; j++){
			for (int i=0; i<nf1; i++){
				printf(" (%2.3g,%2.3g)",fw[i+j*nf1+k*nf2*nf1].real(),
					fw[i+j*nf1+k*nf2*nf1].imag() );
			}
			std::cout<<std::endl;
		}
		std::cout<<"----------------------------------------------------------------"<<std::endl;
	}


	checkCudaErrors(cudaDeviceReset());
	checkCudaErrors(cudaFreeHost(x));
	checkCudaErrors(cudaFreeHost(y));
	checkCudaErrors(cudaFreeHost(z));
	checkCudaErrors(cudaFreeHost(c));
	checkCudaErrors(cudaFreeHost(fw));
	return 0;
}