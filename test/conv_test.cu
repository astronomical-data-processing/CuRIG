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

int main(int argc, char* argv[]){
	//improved WS stacking 1,
	//gpu_method == 0, nupts driven
    int N1, N2;
	PCS sigma = 2.0;
	int M;
	if (argc<5) {
		fprintf(stderr,
			"Usage: conv3d method nupts_distr nf1 nf2 nf3 [maxsubprobsize [M [tol [kerevalmeth [sort]]]]]\n"
			"Arguments:\n"
			"  method: One of\n"
			"    0: nupts driven,\n"
			"    2: sub-problem, or\n"
			"    4: block gather (each nf must be multiple of 8).\n"
            "  w_term_method: \n"
            "    0: w-stacking\n"
            "    1: improved w-stacking\n"
			"  N1, N2 : image size.\n"
			"  M: The number of non-uniform points.\n"
			"  tol: NUFFT tolerance (default 1e-6).\n"
			"  kerevalmeth: Kernel evaluation method; one of\n"
			"     0: Exponential of square root (default), or\n"
			"     1: Horner evaluation.\n"
		);
		return 1;
	}
	//no result
	double w;
	int method;
	sscanf(argv[1],"%d",&method);
	int w_term_method;
	sscanf(argv[2],"%d",&w_term_method);
	sscanf(argv[3],"%lf",&w); N1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[4],"%lf",&w); N2 = (int)w;  // so can read 1e6 right!
	
	M = N1 * N2;
	if(argc>5){
		sscanf(argv[5],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
	}

	PCS tol=1e-6;
	if(argc>6){
		sscanf(argv[6],"%lf",&w); tol  = (PCS)w;  // so can read 1e6 right!
	}

	int kerevalmeth=0;
	if(argc>7){
		sscanf(argv[7],"%d",&kerevalmeth);
	}

    // fov and 1 pixel corresonding to pix_deg degree

	N1 = 5; N2 = 5; M = 25; //for correctness checking
	//int ier;
	PCS *x, *y, *z;
	CPX *c, *fw;
	x = (PCS *)malloc(M*sizeof(PCS)); //Allocates page-locked memory on the host.
	y = (PCS *)malloc(M*sizeof(PCS));
	z = (PCS *)malloc(M*sizeof(PCS));
	c = (CPX *)malloc(M*sizeof(CPX));

	//cudaMallocHost(&fw,nf1*nf2*nf3*sizeof(CPX)); //malloc after plan setting

	PCS *d_x, *d_y, *d_z;
	CUCPX *d_c, *d_fw;
	checkCudaErrors(cudaMalloc(&d_x,M*sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_y,M*sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_z,M*sizeof(PCS)));
	checkCudaErrors(cudaMalloc(&d_c,M*sizeof(CUCPX)));
	//checkCudaErrors(cudaMalloc(&d_fw,8*nf1*nf2*nf1*sizeof(CUCPX)));

    //generating data
    int nupts_distribute = 0;
	switch(nupts_distribute){
		case 0: //uniform
			{
				for (int i = 0; i < M; i++) {
					x[i] = M_PI*randm11();
					y[i] = M_PI*randm11();
					z[i] = M_PI*randm11();
					c[i].real(1); //back to random11()
					c[i].imag(1);
				}
			}
			break;
		case 1: // concentrate on a small region
			{
				for (int i = 0; i < M; i++) {
					x[i] = M_PI*rand01()/N1*16;
					y[i] = M_PI*rand01()/N2*16;
					z[i] = M_PI*rand01()/N2*16;
					c[i].real(randm11());
					c[i].imag(randm11());
				}
			}
			break;
		default:
			std::cerr << "not valid nupts distr" << std::endl;
			return 1;
	}
	x[0] = - PI/2; y[0] = - PI/2; z[0] = - PI/2;
	x[1] = - PI/2; y[1] = - PI/3; z[1] = - PI/2;
	x[2] = - PI/2; y[2] = 0; z[2] = - PI/2;
	x[3] = - PI/2; y[3] = PI/3; z[3] = - PI/2;
	x[4] = - PI/2; y[4] = PI/2; z[4] = - PI/2;

	x[5] = - PI/3; y[5] = - PI/2; z[5] = PI/2;
	x[6] = - PI/3; y[6] = - PI/3; z[6] = PI/2;
	x[7] = - PI/3; y[7] = 0; 	  z[7] = PI/2;
	x[8] = - PI/3; y[8] = PI/3;   z[8] = PI/2;
	x[9] = - PI/3; y[9] = PI/2;   z[9] = PI/2;

	x[10] = 0; y[10] = - PI/2; z[10] = -PI/2;
	x[11] = 0; y[11] = - PI/3; z[11] = -PI/2;
	x[12] = 0; y[12] = 0; 	   z[12] = -PI/2;
	x[13] = 0; y[13] = PI/3;   z[13] = -PI/2;
	x[14] = 0; y[14] = PI/2;   z[14] = -PI/2;

	x[15] = PI/3; y[15] = - PI/2; z[15] = PI/2;
	x[16] = PI/3; y[16] = - PI/3; z[16] = PI/2;
	x[17] = PI/3; y[17] = 0; 	  z[17] = PI/2;
	x[18] = PI/3; y[18] = PI/3;   z[18] = PI/2;
	x[19] = PI/3; y[19] = PI/2;   z[19] = PI/2;

	x[20] = PI/2; y[20] = - PI/2; z[20] = - PI/2;
	x[21] = PI/2; y[21] = - PI/3; z[21] = - PI/2;
	x[22] = PI/2; y[22] = 0; 	  z[22] = - PI/2;
	x[23] = PI/2; y[23] = PI/3;   z[23] = - PI/2;
	x[24] = PI/2; y[24] = PI/2;   z[24] = - PI/2;


	//printf("generated data, x[1] %2.2g, y[1] %2.2g , z[1] %2.2g, c[1] %2.2g\n",x[1] , y[1], z[1], c[1].real());
    //data transfer
	checkCudaErrors(cudaMemcpy(d_x,x,M*sizeof(PCS),cudaMemcpyHostToDevice)); //u
	checkCudaErrors(cudaMemcpy(d_y,y,M*sizeof(PCS),cudaMemcpyHostToDevice)); //v
	checkCudaErrors(cudaMemcpy(d_z,z,M*sizeof(PCS),cudaMemcpyHostToDevice)); //w
	checkCudaErrors(cudaMemcpy(d_c,c,M*sizeof(CUCPX),cudaMemcpyHostToDevice));

    curafft_plan *h_plan = new curafft_plan();
    memset(h_plan, 0, sizeof(curafft_plan));
	
    // opts and copts setting
    h_plan->opts.gpu_conv_only = 1;
    h_plan->opts.gpu_gridder_method = method;
	h_plan->opts.gpu_kerevalmeth = kerevalmeth;
	h_plan->opts.gpu_sort = 1;
	h_plan->opts.upsampfac = sigma;
	// h_plan->copts.pirange = 1;
	// some plan setting
	h_plan->w_term_method = w_term_method;
	

    int ier = setup_conv_opts(h_plan->copts, tol, sigma, kerevalmeth); //check the arguements

	if(ier!=0)printf("setup_error\n");
    
    // plan setting
	
    ier = setup_plan(N1, N2, M, d_x, d_y, d_z, d_c, h_plan); //cautious the number of plane using N1 N2 to get nf1 nf2

	//printf("the num of w %d\n",h_plan->num_w);
	int nf1 = h_plan->nf1;
	int nf2 = h_plan->nf2;
	h_plan->num_w = 2;
	int nf3 = h_plan->num_w; //correctness checking
	printf("the kw is %d\n", h_plan->copts.kw);
	int f_size = nf1*nf2*nf3;
	fw = (CPX *)malloc(sizeof(CPX)*f_size);
	checkCudaErrors(cudaMalloc(&d_fw,f_size*sizeof(CUCPX)));

	h_plan->fw = d_fw;
    //checkCudaErrors(cudaMallocHost(&fw,nf1*nf2*h_plan->num_w*sizeof(CPX))); //malloc after plan setting
    //checkCudaErrors(cudaMalloc( &d_fw,( nf1*nf2*(h_plan->num_w)*sizeof(CUCPX) ) ) ); //check


	std::cout<<std::scientific<<std::setprecision(3);//setprecision not define


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


    // checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(fw,d_fw,sizeof(CUCPX)*f_size,cudaMemcpyDeviceToHost));
	
	//int nf3 = h_plan->num_w;
	printf("Method %d (nupt driven) %ld NU pts to #%d U pts in %.3g s\n",
			h_plan->opts.gpu_gridder_method,M,nf1*nf2*nf3,kernel_time/1000);
	
	
	
		
	
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
	
	checkCudaErrors(cudaFree(d_x));
	checkCudaErrors(cudaFree(d_y));
	checkCudaErrors(cudaFree(d_z));
	checkCudaErrors(cudaFree(d_c));
	checkCudaErrors(cudaFree(d_fw));
	
	checkCudaErrors(cudaDeviceReset());
	free(x);
	free(y);
	free(z);
	free(c);
	free(fw);
	
	return 0;
}