/*
Implementation of Fourier Transfom with uniformly distributed data in frequent domain.

Compare two versions: 1. step by step 2. just one memory transfer. 

1. Direct FT in 1D and 2D

2. Nonuniform Fast Fourier Transform with different kernel for gridding

*/
/*
Remaining problems:
1. get maximum threads
2. 
*/
//#include <iostream>
//#include <cstdio>

#include "dft.cuh"


#define CHECK(call)                                                     \
    {                                                                   \
        const cudaError_t error = call;                                 \
        if (error != cudaSuccess)                                       \
        {                                                               \
            printf("Error:%s:%d", __FILE__, __LINE__);                  \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString); \
            exit(1);                                                    \
        }                                                               \
    }

__global__ void freqs_kernel(int M, float df, int *ind, bool smt)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int offset = 0;
    if (smt == true)
    {
        offset = - M / 2;
    }

    while (idx < M)
    {
        ind[idx] = idx + offset;
        idx = idx + blockDim.x * gridDim.x;
    }
}

void freqs(int *array, int M, float df = 1.0, bool smt = true)
{
    /*
        Create a range of frequencies
        
        Inputs:
            M - size of output (the height or width of image)
            df - scaling size
            smt - symmetry
        Ouputs:
            an array with grid frequencies.
    */

    int *A_d;
    /*----------Memory allocaiton-----------*/
    cudaMalloc((void **)&A_d, M * sizeof(int));

    /*----------Invoke kernel-----------*/
    dim3 block(THREADNUM);
    dim3 grid((M-1)/THREADNUM+1); //get a optimal size

    freqs_kernel<<<grid, block>>>(M, df, A_d, smt);
    cudaDeviceSynchronize();
    /*----------Data Transfer-----------*/
    cudaMemcpy(array, A_d, M * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
}



void directFT_1d_m(int M, complex<float> *c, float *x, int length, int direction, float df = 1.0)
{
    //the type of c shoule not be fixed
    /*
        Direct Fourier Transform in 1d

        Inputs:
            c - complex number, fourier coefficient
            x - real location, frequency related
            direction - CUFFTFORWARD, CUFFTINVERSE
        Outputs:
            complex number
    */
    int N = length;
    /*----------Rescaling-----------*/
    if (df != 1.0)
    {
        for (int i = 0; i < length; i++)
        {
            x[i] = x[i] * df;
        }
    }
    int *k = (int *)malloc(M * sizeof(int));
    freqs(k, M);
    //std::cout<<"freqs fine"<<'\n';
    int sign = 0;
    float scale_ratio = 1.0;
    if (direction == CUFFT_FORWARD)
    {
        sign = -1;
    }
    else if (direction == CUFFT_INVERSE)
    {
        sign = 1;
        scale_ratio = 1.0 / float(N);
    }
    complex<float> unit = 1.0if;
    //float * temp_result1 = (float *)malloc(sizeof(float)*length*M); // real number or complex number...
    complex<float>* temp_result = (complex<float>*)malloc(sizeof(complex<float>)*M*length);
    Boardcast<int,float,complex<float>>(k,x,temp_result,M,length);
    //std::cout<<"Boardcast fine"<<'\n';
    
    matrix_elementwise_operation<complex<float>>(temp_result, sign*unit, temp_result, int('*'), length, M);
    matrix_elementwise_operation<complex<float>>(temp_result, temp_result, EXP, length, M);
    //std::cout<<"elem fine"<<'\n';
    complex<float> *result = (complex<float> *)malloc(sizeof(complex<float>)*M);
    MatrixMulMatrix<complex<float>>(c, temp_result, result, 1, length, M);
    //std::cout<<"mul fine"<<'\n';
    matrix_elementwise_operation<complex<float>>(result,scale_ratio,result,int('*'), 1, M);

    for(int i=0;i<M;i++){
        std::cout<<result[i]<<" ";
    }
    std::cout<<"\n";

}

// DFT With once memory transfer
void directFT_1d(int M, complex<float> *c, float *x, int length, int direction, float df = 1.0){
    //the type of c should not be fixed
    /*
        Direct Fourier Transform in 1d

        Inputs:
            c - complex number, fourier coefficient
            x - real location, frequency related
            direction - CUFFTFORWARD -1, CUFFTINVERSE 1
        Outputs:
            complex number
    */
    int N = length;
    /*----------Rescaling-----------*/
    if (df != 1.0)
    {
        for (int i = 0; i < length; i++)
        {
            x[i] = x[i] * df;
        }
    }
    int *k = (int *)malloc(M * sizeof(int));
    freqs(k, M);
    
    //std::cout<<"freqs fine"<<'\n';
    int sign = 0;
    float scale_ratio = 1.0;
    if (direction == CUFFT_FORWARD)
    {
        sign = -1;
    }
    else if (direction == CUFFT_INVERSE)
    {
        sign = 1;
        scale_ratio = 1.0 / float(N);
    }
    complex<float> unit = 1.0if;
    //std::cout<<CUFFT_FORWARD<<" "<<sign<<"\n";
    complex<float>* h_result = (complex<float>*)malloc(sizeof(complex<float>)*M);
    
    int *d_v1;  //vector of k in device, d_k
    float *d_v2; //vector of x
    complex<float> *d_c; //c in device
    complex<float> *d_t_ret; //temporate result in device
    complex<float> *d_result; //result


    cudaMalloc((void **)&d_result, sizeof(complex<float>) * M);
    cudaMalloc((void **)&d_v1, sizeof(int) * M);
    cudaMalloc((void **)&d_v2, sizeof(float) * length);
    cudaMalloc((void **)&d_t_ret, sizeof(complex<float>) * M * length);
    cudaMalloc((void **)&d_c, sizeof(complex<float>) *  length);
    cudaMemset((void **)&d_result, 0, sizeof(complex<float>) * M);
    
    cudaMemcpy(d_v1, k, sizeof(int) * M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, x, sizeof(float) * length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, sizeof(complex<float>) * length, cudaMemcpyHostToDevice);

    dim3 block(THREADNUM, THREADNUM);
    dim3 grid((M-1)/THREADNUM+1,(length-1)/THREADNUM+1);

    /*----------invoke kernel----------*/
    Boardcast_kernel<int,float,complex<float>><<<grid, block>>>(d_v1, d_v2, d_t_ret, M, length);
    cudaDeviceSynchronize();
    matrix_elementwise_operation_kernel<complex<float>><<<grid, block>>>(d_t_ret, sign*unit, d_t_ret, int('*'), length, M);
    cudaDeviceSynchronize();
    matrix_elementwise_operation_kernel<complex<float>><<<grid, block>>>(d_t_ret, d_t_ret, EXP, length, M);
    cudaDeviceSynchronize();
    


    

    //reconsider the grid size.
    dim3 grid_1((int)ceil(M * 1.0 / Tile_Width), (int)ceil(1 * 1.0 / Tile_Width));
    dim3 block_1(Tile_Width, Tile_Width);
    MatrixMulMatrix_kernel2<complex<float>><<<grid_1, block_1>>>(d_c, d_t_ret, d_result, 1, length, M);
    CHECK(cudaDeviceSynchronize());


    //cudaMemcpy(h_result, d_result, sizeof(complex<float>) * M, cudaMemcpyDeviceToHost);
    

    matrix_elementwise_operation_kernel<complex<float>><<<grid_1, block_1>>>(d_result,scale_ratio,d_result,int('*'), 1, M);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_result, sizeof(complex<float>) * M, cudaMemcpyDeviceToHost);
    for(int i=0;i<M;i++){
        std::cout<<h_result[i]<<" ";
    }
    std::cout<<"\n";
    
    
    cudaFree(d_result);
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_c);
    cudaFree(d_t_ret);
    

    free(h_result);
    free(k);

}

void directFT_2d(int M, int N, complex<float> *c, float *u, float *v, int length, int direction, float df = 1.0){
    /*
        Direct Fourier Transform in 2d

        Inputs:
            (M,N) - size of output
            c - complex number, fourier coefficient, visibility  
            u,v - real location, frequency related.     (u,v,c)[i], range of u (0,2*pi) or (-pi,pi)
            length - total number of visibility or c
            df - scaling ratio to keep uv at proper range//should have two for each dimension
            direction - CUFFTFORWARD -1, CUFFTINVERSE 1
        Outputs:
            complex number
    */
    
    /*----------Rescaling-----------*/
    //should have a kernel for this operation, keep uv at (-pi,pi)
    if (df != 1.0)
    {
        for (int i = 0; i < length; i++)
        {
            u[i] = u[i] * df;
            v[i] = v[i] * df;
        }
    }
    int *array_m = (int *)malloc(M * N * sizeof(int));
    int *array_n = (int *)malloc(N * M * sizeof(int));
    //need to revise
    //write a 2d freqs and need a proper name
    freqs(array_m,M*N);
    freqs(array_n,N*M);
    
    //std::cout<<"freqs fine"<<'\n';
    int sign = 0;
    float scale_ratio = 1.0;
    if (direction == CUFFT_FORWARD)
    {
        sign = -1;
    }
    else if (direction == CUFFT_INVERSE)
    {
        sign = 1;
        scale_ratio = 1.0 / float(length);
    }
    complex<float> unit = 1.0if;
    //std::cout<<CUFFT_FORWARD<<" "<<sign<<"\n";
    complex<float>* h_result = (complex<float>*)malloc(sizeof(complex<float>)*M*N); //finial result in host
    
    int *d_m;  //vector of m
    int *d_n;  //len = M*N
    float *d_u; //vector of u
    float *d_v;
    complex<float> *d_c; //c in device
    complex<float> *d_t_ret_v, *d_t_ret_u; 
    complex<float> *d_t_ret; //temporate result in device
    complex<float> *d_result; //result


    cudaMalloc((void **)&d_result, sizeof(complex<float>) * M*N);
    cudaMalloc((void **)&d_m, sizeof(int) * M*N);
    cudaMalloc((void **)&d_n, sizeof(int) * M*N);
    cudaMalloc((void **)&d_u, sizeof(float) * length);
    cudaMalloc((void **)&d_v, sizeof(float) * length);
    cudaMalloc((void **)&d_t_ret_u, sizeof(complex<float>) * M * N * length); //R
    cudaMalloc((void **)&d_t_ret_v, sizeof(complex<float>) * M * N * length); //R
    cudaMalloc((void **)&d_t_ret, sizeof(complex<float>) * M * N * length); //R
    cudaMalloc((void **)&d_c, sizeof(complex<float>) *  length);
    cudaMemset((void **)&d_result, 0, sizeof(complex<float>) * M *N);
    
    cudaMemcpy(d_m, array_m, sizeof(int) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, array_n, sizeof(int) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, sizeof(float) * length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, sizeof(float) * length, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, sizeof(complex<float>) * length, cudaMemcpyHostToDevice);

    dim3 block(THREADNUM, THREADNUM);
    dim3 grid((M*N-1)/THREADNUM+1,(length-1)/THREADNUM+1);

    /*----------invoke kernel----------*/
    Boardcast_kernel<int,float,complex<float>><<<grid, block>>>(d_m, d_u, d_t_ret_u, M*N, length); //revise to float
    cudaDeviceSynchronize();
    Boardcast_kernel<int,float,complex<float>><<<grid, block>>>(d_n, d_v, d_t_ret_v, M*N, length);
    cudaDeviceSynchronize();
    //float float complex
    matrix_elementwise_operation_kernel<complex<float>><<<grid, block>>>(d_t_ret_u, d_t_ret_v, d_t_ret, int('+'), length, M*N);
    cudaDeviceSynchronize();

    cudaFree(d_t_ret_v);
    cudaFree(d_t_ret_u); 

    matrix_elementwise_operation_kernel<complex<float>><<<grid, block>>>(d_t_ret, sign*unit, d_t_ret, int('*'), length, M*N);
    cudaDeviceSynchronize();
    matrix_elementwise_operation_kernel<complex<float>><<<grid, block>>>(d_t_ret, d_t_ret, EXP, length, M);
    cudaDeviceSynchronize();
    


    

    //reconsider the grid size.
    dim3 grid_1((int)ceil(M*N * 1.0 / Tile_Width), (int)ceil(1 * 1.0 / Tile_Width));
    dim3 block_1(Tile_Width, Tile_Width);
    MatrixMulMatrix_kernel2<complex<float>><<<grid_1, block_1>>>(d_c, d_t_ret, d_result, 1, length, M*N);
    CHECK(cudaDeviceSynchronize());


    //cudaMemcpy(h_result, d_result, sizeof(complex<float>) * M*N, cudaMemcpyDeviceToHost);
    

    matrix_elementwise_operation_kernel<complex<float>><<<grid_1, block_1>>>(d_result,scale_ratio,d_result,int('*'), 1, M*N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_result, sizeof(complex<float>) * M*N, cudaMemcpyDeviceToHost);
    for(int i=0;i<M*N;i++){
        std::cout<<h_result[i]<<" ";
    }
    std::cout<<"\n";
    
    
    cudaFree(d_result);
    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_c);
    cudaFree(d_t_ret);
    

    free(h_result);
    free(array_m);
    free(array_n);
}

int main(){
    
    float * x = (float*)malloc(sizeof(float)*20);
    complex<float> *c = (complex<float> *)malloc(sizeof(complex<float>)*20);
    srand(1221);
    for(int i=0; i<20; i++){
        //x[i] = rand() / float(RAND_MAX) * 2000;
        x[i] = i * 100.0;
        c[i] = exp(1.0if*x[i]);
    }
    directFT_1d( 11,  c, x, 20, 1);
    
    return 0;
}