/*
convolutional kernel
hliuco
date 03 October
Methods:
1. NUPT Drive
2. UPT Drive
Dimensions:
1,2,3
*/

///issue: the result contain 0.000 and ends with 8.000 and %

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/curafft_plan.h"

/*
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
*/
//one dimensional 1024, 2 dimensional 32
#define KERNEL_WIDTH 3
#define BLOCK_WIDTH  8
#define TILE_WIDTH (BLOCK_WIDTH-KERNEL_WIDTH+1)

//w-term conv
// M is larger than total number of threads

__global__ void conv_1D(float *N, float *M, float *P, int Width){
    //Width is the input array width and height
	int idx = threadIdx.x;
	int row_c = blockIdx.x*TILE_WIDTH + idx;	
	int row_l = row_c - KERNEL_WIDTH/2;
	
	__shared__ int s_input[BLOCK_WIDTH];
	if (row_l>=0&&row_l<Width){
		s_input[idx] = N[row_l];
	}
	else{
		s_input[idx]=0.0f;
	}
	__syncthreads();

	float pvalue=0;
	if(idx<TILE_WIDTH){
		for(int i=0; i < KERNEL_WIDTH; i++){
			pvalue += M[i]*s_input[idx+i];
		}
	}

	if(row_c<Width) P[row_c] = pvalue;
}

__global__ void conv_2D(float *N, float *M, float *P, int Width){
    //Width is the input array width
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	int col_c = blockIdx.x*TILE_WIDTH + idx;
    int row_c = blockIdx.y*TILE_WIDTH + idy;	
	int row_l = row_c - KERNEL_WIDTH/2;
	int col_l = col_c - KERNEL_WIDTH/2; 
    //if(idx==10&&idy==10)printf("blockidx %d, blockidy %d, row_l %d, col_l %d\n",row_l,col_l);
	//if(idx == 0)printf("1 \n");
	__shared__ float s_input[BLOCK_WIDTH][BLOCK_WIDTH];
	if (row_l>=0&&col_l>=0&&row_l<Width&&col_l<Width){
		s_input[idy][idx] = N[row_l*Width + col_l];
	    //printf("%.3lf, (%d,%d) \n",s_input[idy][idx],row_l,col_l);
	}
	else{
		s_input[idy][idx]=0.0f;
	}
	__syncthreads();

	float pvalue=0;
	if(idx<TILE_WIDTH&&idy<TILE_WIDTH){
		for(int i=0; i < KERNEL_WIDTH; i++){
			for(int j=0; j<KERNEL_WIDTH; j++){
				pvalue += M[i*KERNEL_WIDTH+j] * s_input[idy+i][idx+j];
			}
		}
	}
    
	if(row_c<Width&&col_c<Width) P[row_c*Width+col_c] = pvalue;
}

__global__ void conv_3D(float *N, float *M, float *P, int Width){
    //Width is the input array width in i dimension
	int idx = threadIdx.x;
	int idy = threadIdx.y;
    int idz = threadIdx.z;
	int col_c = blockIdx.x*TILE_WIDTH + idx;
    int row_c = blockIdx.y*TILE_WIDTH + idy;
    int chan_c = blockIdx.z*TILE_WIDTH + idz;	
	int row_l = row_c - KERNEL_WIDTH/2;
	int col_l = col_c - KERNEL_WIDTH/2;
    int chan_l = chan_c - KERNEL_WIDTH/2;
    //if(idx==10&&idy==10)printf("row_l %d, col_l %d, chan_l %d\n",row_l,col_l,chan_l);
	//if(idx == 0)printf("1 \n");
	__shared__ float s_input[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];
	if (row_l>=0&&col_l>=0&&chan_l>0&&row_l<Width&&col_l<Width&&chan_l<Width){
		s_input[idz][idy][idx] = N[chan_l*Width*Width + row_l*Width + col_l];
	    //printf("%.3lf, (%d,%d) \n",s_input[idz][idy][idx],row_l,col_l);
	}
	else{
		s_input[idz][idy][idx]=0.0f;
	}
	__syncthreads();

	float pvalue=0;
	if(idx<TILE_WIDTH&&idy<TILE_WIDTH&&idz<TILE_WIDTH){
		for(int i=0; i < KERNEL_WIDTH; i++){
			for(int j=0; j<KERNEL_WIDTH; j++){
                for(int k=0; k<KERNEL_WIDTH; k++){
				    pvalue += M[i*KERNEL_WIDTH*KERNEL_WIDTH+j*KERNEL_WIDTH+k] * s_input[idz+i][idy+j][idx+k];
                }
			}
		}
	}
    
	if(row_c<Width&&col_c<Width&&row_c<Width) P[chan_c*Width*Width+row_c*Width+col_c] = pvalue;
}

/*
in order to using shared memory,the block size = tilewidth + 
mask_width -1
some threads just for loading shared memory, tile threads for calculation.
*/


int main(){

	float *h_N,*h_M,*h_P;

	float *d_N,*d_M,*d_P;
    int dim = 3;
    //test 2d
    //BLOCK_SIZE = 32 KERNEL_WIDTH = 5
    if(dim == 2)
	{
        int size = 4096;

        int byte_size = size*sizeof(float);
        
        h_N = (float *)malloc(byte_size);
        h_M = (float *)malloc(KERNEL_WIDTH*KERNEL_WIDTH*sizeof(float));
        h_P = (float *)malloc(byte_size);

        for(int i=0; i<size; i++){
            h_N[i] = 1;
        }
        for(int i=0; i< KERNEL_WIDTH*KERNEL_WIDTH; i++){
            h_M[i] = 1;
        }

        CHECK(cudaMalloc((void **)&d_N,byte_size));
        CHECK(cudaMalloc((void **)&d_M,KERNEL_WIDTH*KERNEL_WIDTH*sizeof(float)));
        CHECK(cudaMalloc((void **)&d_P,byte_size));

        CHECK(cudaMemcpy(d_N,h_N,byte_size,cudaMemcpyHostToDevice));
        cudaMemcpy(d_M,h_M,KERNEL_WIDTH*KERNEL_WIDTH*sizeof(float),cudaMemcpyHostToDevice);

        dim3 block(BLOCK_WIDTH,BLOCK_WIDTH);
        dim3 grid((64-1)/TILE_WIDTH+1,(64-1)/TILE_WIDTH+1);

        conv_2D<<<grid,block>>>(d_N,d_M,d_P,64);

        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_P,d_P,byte_size,cudaMemcpyDeviceToHost));

        for(int i=0; i<size; i++){printf("%lf ", h_P[i]);}

        free(h_M);
        free(h_N);
        free(h_P);
        cudaFree(d_P);
        cudaFree(d_M);
        cudaFree(d_N);
    }
    //3d
    //BLOCK_SIZE = 8 KERNEL_WIDTH = 3
    if(dim == 3){
        int size = 4096;

        int byte_size = size*sizeof(float);
        
        h_N = (float *)malloc(byte_size);
        h_M = (float *)malloc(KERNEL_WIDTH*KERNEL_WIDTH*KERNEL_WIDTH* sizeof(float));
        h_P = (float *)malloc(byte_size);

        for(int i=0; i<size; i++){
            h_N[i] = 1;
        }
        for(int i=0; i< KERNEL_WIDTH*KERNEL_WIDTH*KERNEL_WIDTH; i++){
            h_M[i] = 1;
        }

        CHECK(cudaMalloc((void **)&d_N,byte_size));
        CHECK(cudaMalloc((void **)&d_M,KERNEL_WIDTH*KERNEL_WIDTH*KERNEL_WIDTH* sizeof(float)));
        CHECK(cudaMalloc((void **)&d_P,byte_size));

        CHECK(cudaMemcpy(d_N,h_N,byte_size,cudaMemcpyHostToDevice));
        cudaMemcpy(d_M,h_M,KERNEL_WIDTH*KERNEL_WIDTH*KERNEL_WIDTH* sizeof(float),cudaMemcpyHostToDevice);

        dim3 block(BLOCK_WIDTH,BLOCK_WIDTH,BLOCK_WIDTH);
        dim3 grid((16-1)/TILE_WIDTH+1,(16-1)/TILE_WIDTH+1,(16-1)/TILE_WIDTH+1);

        conv_3D<<<grid,block>>>(d_N,d_M,d_P,16);

        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(h_P,d_P,byte_size,cudaMemcpyDeviceToHost));

        for(int i=0; i<size; i++){printf("%lf ", h_P[i]);}

        free(h_M);
        free(h_N);
        free(h_P);
        cudaFree(d_P);
        cudaFree(d_M);
        cudaFree(d_N);
    }
}