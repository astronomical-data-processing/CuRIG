#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <iostream>
#include <thrust/complex.h>
//using namespace std;
using namespace thrust;
using namespace std::complex_literals;
#define THREADNUM 32
#define ABS -1
#define EXP -2
#define Tile_Width 32
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


template <typename T1, typename T2, typename T3>
__global__ void Boardcast_kernel(T1 *d_v1, T2 *d_v2, T3 *M, int n1, int n2)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int temp = idx;
    while (idy < n2)
    {
        while (idx < n1)
        {
            M[idy * n1 + idx] = d_v1[idx] * d_v2[idy];
            idx = idx + gridDim.x * blockDim.x;
        }
        idx = temp;
        idy = idy + gridDim.y * blockDim.y;
    }
}

template <typename T1, typename T2, typename T3>
void Boardcast(T1 *h_v1, T2 *h_v2, T3 *M, int n1, int n2)
{
    /*
        get a matrix with shape (n2,n1)
        Inputs:
            h_v1 - row vector with shape (1,n1)
            h_v2 - column vector with shape (n2,1)
        Outputs:
            M - matrix after boardcasting with shape (n2,n1)
    */
    T1 *d_v1;
    T2 *d_v2;
    T3 *d_M;

    cudaMalloc((void **)&d_v1, sizeof(T1) * n1);
    cudaMalloc((void **)&d_v2, sizeof(T2) * n2);
    cudaMalloc((void **)&d_M, sizeof(T3) * n1 * n2);

    cudaMemcpy(d_v1, h_v1, sizeof(T1) * n1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, h_v2, sizeof(T2) * n2, cudaMemcpyHostToDevice);

    dim3 block(THREADNUM, THREADNUM);
    dim3 grid((n1 - 1) / THREADNUM + 1, (n2 - 1) / THREADNUM + 1);

    /*----------invoke kernel----------*/
    Boardcast_kernel<T1, T2, T3><<<grid, block>>>(d_v1, d_v2, d_M, n1, n2);
    cudaDeviceSynchronize();

    cudaMemcpy(M, d_M, sizeof(T3) * n1 * n2, cudaMemcpyDeviceToHost);

    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_M);
}

template <typename T>
__global__ void matrix_elementwise_operation_kernel(T *M, T *N, T *ret, int op, int m, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int index = 0;
    switch (op)
    {
    case '+':
    {
        while (idy < m)
        {
            while (idx < n)
            {
                index = idy * n + idx; //caution
                ret[index] = M[index] + N[index];
                idx = idx + blockDim.x * gridDim.x;
            }
            idx = threadIdx.x + blockIdx.x * blockDim.x;
            idy = idy + blockDim.y * gridDim.y;
        }
        break;
    };
    case '-':
    {
        while (idy < m)
        {
            while (idx < n)
            {
                index = idy * n + idx; //caution
                ret[index] = M[index] - N[index];
                idx = idx + blockDim.x * gridDim.x;
            }
            idx = threadIdx.x + blockIdx.x * blockDim.x;
            idy = idy + blockDim.y * gridDim.y;
        }
        break;
    };
    case '*':
    {
        while (idy < m)
        {
            while (idx < n)
            {
                index = idy * n + idx; //caution
                ret[index] = M[index] * N[index];
                idx = idx + blockDim.x * gridDim.x;
            }
            idx = threadIdx.x + blockIdx.x * blockDim.x;
            idy = idy + blockDim.y * gridDim.y;
        }
        break;
    };
    case '/':
    {
        while (idy < m)
        {
            while (idx < n)
            {
                index = idy * n + idx; //caution
                ret[index] = M[index] / N[index];
                idx = idx + blockDim.x * gridDim.x;
            }
            idx = threadIdx.x + blockIdx.x * blockDim.x;
            idy = idy + blockDim.y * gridDim.y;
        }
        break;
    };
    default:
    {
    }
    }
}

template <typename T>
void matrix_elementwise_operation(T *M, T *N, T *ret, int op, int m, int n)
{
    /*
        Matrix related elementwise operation. GPU version.
        Inputs:
            M - matrix M, shape(m,n)
            N - matrix N, shape(m,n)
            op - operation type 
        Outputs:
            ret - result
    */
    T *d_M, *d_N, *d_ret;

    int nBytes = sizeof(T) * m * n;
    cudaMalloc((void **)&d_M, nBytes);
    cudaMalloc((void **)&d_N, nBytes);
    cudaMalloc((void **)&d_ret, nBytes);

    cudaMemcpy(d_M, M, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, nBytes, cudaMemcpyHostToDevice);

    dim3 block(THREADNUM, THREADNUM);
    dim3 grid((n - 1) / THREADNUM + 1, (m - 1) / THREADNUM + 1); //get optimal size, first element maps to gridDim.x, second maps to gridDim.y

    matrix_elementwise_operation_kernel<T><<<grid, block>>>(d_M, d_N, d_ret, op, m, n);
    cudaDeviceSynchronize();
    cudaMemcpy(ret, d_ret, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_ret);
    cudaFree(d_M);
    cudaFree(d_N);
}

template <typename T>
__global__ void matrix_elementwise_operation_kernel(T *M, T elem, T *ret, int op, int m, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int index = idy * n + idx;
    __shared__ T s_elem;
    int l_id = threadIdx.x * threadIdx.y; //threads take charge of load data to shared memeory.
    if (l_id == 0)
    {
        s_elem = elem;
    }
    __syncthreads();
    switch (op)
    {
    case '+':
    {
        while (idy < m)
        {
            while (idx < n)
            {
                index = idy * n + idx; //caution
                ret[index] = M[index] + s_elem;
                idx = idx + blockDim.x * gridDim.x;
            }
            idx = threadIdx.x + blockIdx.x * blockDim.x;
            idy = idy + blockDim.y * gridDim.y;
        }
        break;
    };
    case '-':
    {
        while (idy < m)
        {
            while (idx < n)
            {
                index = idy * n + idx; //caution
                ret[index] = M[index] - s_elem;
                idx = idx + blockDim.x * gridDim.x;
            }
            idx = threadIdx.x + blockIdx.x * blockDim.x;
            idy = idy + blockDim.y * gridDim.y;
        }
        break;
    };
    case '*':
    {
        while (idy < m)
        {
            while (idx < n)
            {
                index = idy * n + idx; //caution
                ret[index] = M[index] * s_elem;
                idx = idx + blockDim.x * gridDim.x;
            }
            idx = threadIdx.x + blockIdx.x * blockDim.x;
            idy = idy + blockDim.y * gridDim.y;
        }
        break;
    };
    case '/':
    {
        while (idy < m)
        {
            while (idx < n)
            {
                index = idy * n + idx; //caution
                ret[index] = M[index] / s_elem;
                idx = idx + blockDim.x * gridDim.x;
            }
            idx = threadIdx.x + blockIdx.x * blockDim.x;
            idy = idy + blockDim.y * gridDim.y;
        }
        break;
    };
    default:
    {
    }
    }
}

template <typename T>
void matrix_elementwise_operation(T *M, T elem, T *ret, int op, int m, int n)
{
    /*
        Matrix related elementwise operation. GPU version. e.g. 2 * M
        Inputs:
            M - matrix M, shape(m,n)
            elem - single element
            op - operation type 
        Outputs:
            ret - result
    */
    T *d_M, *d_ret;

    int nBytes = sizeof(T) * m * n;
    cudaMalloc((void **)&d_M, nBytes);

    cudaMalloc((void **)&d_ret, nBytes);

    cudaMemcpy(d_M, M, nBytes, cudaMemcpyHostToDevice);

    dim3 block(THREADNUM, THREADNUM);
    dim3 grid((n - 1) / THREADNUM + 1, (m - 1) / THREADNUM + 1); //get optimal size, first element maps to gridDim.x, second maps to gridDim.y

    matrix_elementwise_operation_kernel<T><<<grid, block>>>(d_M, elem, d_ret, op, m, n);
    cudaDeviceSynchronize();
    cudaMemcpy(ret, d_ret, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_ret);
    cudaFree(d_M);
}

template <typename T>
__global__ void matrix_elementwise_operation_kernel(T *M, T *ret, int op, int m, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int index = 0;

    switch (op)
    {
    case EXP:
    {
        while (idy < m)
        {
            while (idx < n)
            {
                index = idy * n + idx; //caution
                ret[index] = exp(M[index]);
                idx = idx + blockDim.x * gridDim.x;
            }
            idx = threadIdx.x + blockIdx.x * blockDim.x;
            idy = idy + blockDim.y * gridDim.y;
        }
        break;
    };
    case ABS:
    {
        while (idy < m)
        {
            while (idx < n)
            {
                index = idy * n + idx; //caution
                ret[index] = abs(M[index]);
                idx = idx + blockDim.x * gridDim.x;
            }
            idx = threadIdx.x + blockIdx.x * blockDim.x;
            idy = idy + blockDim.y * gridDim.y;
        }
        break;
    };
    default:
    {
    }
    }
}

template <typename T>
void matrix_elementwise_operation(T *M, T *ret, int op, int m, int n)
{
    /*
        Matrix related elementwise operation. GPU version. e.g. exp(M)
        Inputs:
            M - matrix M, shape(m,n)
            op - operation type 
        Outputs:
            ret - result
    */
    T *d_M, *d_ret;

    int nBytes = sizeof(T) * m * n;
    cudaMalloc((void **)&d_M, nBytes);

    cudaMalloc((void **)&d_ret, nBytes);

    cudaMemcpy(d_M, M, nBytes, cudaMemcpyHostToDevice);

    dim3 block(THREADNUM, THREADNUM);
    dim3 grid((n - 1) / THREADNUM + 1, (m - 1) / THREADNUM + 1); //get optimal size, first element maps to gridDim.x, second maps to gridDim.y

    matrix_elementwise_operation_kernel<T><<<grid, block>>>(d_M, d_ret, op, m, n);
    cudaDeviceSynchronize();
    cudaMemcpy(ret, d_ret, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_ret);
    cudaFree(d_M);
}

template <typename T>
__global__ void MatrixMulMatrix_kernel(T *d_M, T *d_N, T *d_P, int m, int k, int n)
{
    /*
    This is general case of multiplication, do not consider DFT specially.
    */
    __shared__ T ds_M[Tile_Width][Tile_Width];
    __shared__ T ds_N[Tile_Width][Tile_Width];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * Tile_Width + ty;
    int col = bx * Tile_Width + tx;

    while (row < m)
    {
        while (col < n)
        {
            // handling the size of output is out of the limit of maximum threads number.

            T resultT = 0;

            for (int i = 0; i < ((k - 1) / Tile_Width + 1); i++)
            {
                // load the Tiles into shared memory

                int m_col = i * Tile_Width + tx;
                int n_row = i * Tile_Width + ty;
                if (m_col < k && row < m)
                {
                    ds_M[ty][tx] = d_M[row * k + i * Tile_Width + tx];
                }
                else
                {
                    ds_M[ty][tx] = 0;
                }
                if (n_row < k && col < n)
                {
                    ds_N[ty][tx] = d_N[col + (i * Tile_Width + ty) * n];
                }
                else
                    ds_N[ty][tx] = 0;
                __syncthreads(); //maybe inproper.

                for (int j = 0; j < Tile_Width; j++)
                {
                    resultT += ds_M[ty][j] * ds_N[j][tx];
                }
                __syncthreads();
            }

            if (row < m && col < n)
            {
                d_P[row * n + col] = resultT;
            }
            __syncthreads();

            bx = bx + gridDim.x;
            row = by * Tile_Width + ty;
            col = bx * Tile_Width + tx;
        }
        bx = blockIdx.x;
        by = by + gridDim.y;
        row = by * Tile_Width + ty;
        col = bx * Tile_Width + tx;
    }
}

template <typename T>
__global__ void MatrixMulMatrix_kernel2(T *d_M, T *d_N, T *d_P, int m, int k, int n)
{
    __shared__ T ds_M[Tile_Width][Tile_Width];
    __shared__ T ds_N[Tile_Width][Tile_Width];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * Tile_Width + ty;
    int col = bx * Tile_Width + tx;
    

    T resultT = 0;
    // in this method we do not consider the size of output is out of the limit of maximum threads number.
    for (int i = 0; i < ((k - 1) / Tile_Width + 1); i++)
    {

        // load the Tiles into shared memory
        int m_col = i * Tile_Width + tx;
        int n_row = i * Tile_Width + ty;
        
        if (m_col < k && row < m)
        {
            ds_M[ty][tx] = d_M[row * k + i * Tile_Width + tx];
        }
        else
        {
            ds_M[ty][tx] = 0;
        }
        if (n_row < k && col < n)
        {
            ds_N[ty][tx] = d_N[col + (i * Tile_Width + ty) * n];
        }
        else
            ds_N[ty][tx] = 0;
        __syncthreads(); //maybe inproper maybe if should end before this command. check with print in the last.

        for (int j = 0; j < Tile_Width; j++)
        {
            resultT += ds_M[ty][j] * ds_N[j][tx];
            //printf("I am thread %d, value is %lf\n",tx,resultT.real());
        }
        __syncthreads();
    }

    if (row < m && col < n)
    {
        d_P[row * n + col] = resultT;
    }
    
}

template <typename T>
void MatrixMulMatrix(T *h_M, T *h_N, T *h_P, int m, int k, int n)
{
    /*
        Tailed Matrix multiplication.
        Inputs:
            M,N - array, two matrices, shape - (m,k), (k,n)
        Outputs:
            P - array, matrix,  shape - (m,n), P = M * N
             

    */
    //invoke and need to refine for complex number.
    T *d_M, *d_N, *d_P;

    cudaMalloc((void **)&d_M, sizeof(T) * m * k);
    cudaMalloc((void **)&d_N, sizeof(T) * k * n);
    cudaMalloc((void **)&d_P, sizeof(T) * m * n);
    cudaMemset((void **)&d_P, 0, sizeof(T) * m * n);

    cudaMemcpy(d_M, h_M, sizeof(T) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, sizeof(T) * k * n, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //reconsider the grid size.
    dim3 grid((int)ceil(n * 1.0 / Tile_Width), (int)ceil(m * 1.0 / Tile_Width));
    dim3 block(Tile_Width, Tile_Width);
    
    //cudaDeviceSynchronize();
    MatrixMulMatrix_kernel2<T><<<grid, block>>>(d_M, d_N, d_P, m, k, n);

    cudaEventRecord(stop, 0);
    CHECK(cudaDeviceSynchronize());
    cudaEventSynchronize(stop);
    float ElapsedTime;
    cudaEventElapsedTime(&ElapsedTime, start, stop);
    printf("Kernel Elpased Time: %.3f ms\n", ElapsedTime);

    cudaMemcpy(h_P, d_P, m * n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

#endif