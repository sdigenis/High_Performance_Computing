#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h" 

#define NUM_BANKS 32 
#define LOG_NUM_BANKS 5 
#define CONFLICT_FREE_OFFSET(n)\
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

#define CUDA_CHECK_ERROR(value) CheckCudaError(__FILE__,__LINE__, #value, value)

static void CheckCudaError(const char *file, unsigned line, const char *statement, cudaError_t error)
{
	if (error == cudaSuccess)
		return;
	printf("CUDA Error: %s\n", cudaGetErrorString(error));
    cudaDeviceReset();
	exit (1);
}

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin, int width, int height){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < height; i ++){
        for(int j = 0; j < width; j++){
                hist_out[img_in[i * width + j]] ++;
        }  
    }
}

__constant__ int c_hist[256];
__constant__ int c_lut[256];

__global__ void histogram(int width, int height, int *d_hist, unsigned char * d_img_in){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int d = d_img_in[y * width + x];
    atomicAdd((unsigned int *)&d_hist[d], 1);

}

__global__ void prefix_sum(int * lut, int img_size){

    __shared__ int d_cdf[256];
    int idx = threadIdx.x;

    int idx2 = threadIdx.x + 128;
    int bankOffsetA = CONFLICT_FREE_OFFSET(idx);
    int bankOffsetB = CONFLICT_FREE_OFFSET(idx2);

    d_cdf[idx + bankOffsetA] = c_hist[idx];
    d_cdf[idx2 + bankOffsetB] = c_hist[idx2];

    int i = 0, d, min = 0;

    while(min == 0){
        min = c_hist[i++]; 
    }
    d = img_size - min;

    //calculate d_cdf with Blelloch algorithm

    #pragma omp unroll
    for(i = 1; i <= 128; i *= 2){
        __syncthreads();
        if((idx + 1)  % i == 0){
            int ai = 2 * idx + 1 - i;
            //ai += CONFLICT_FREE_OFFSET(ai);
            int bi = 2 * idx + 1;
            //bi += CONFLICT_FREE_OFFSET(bi);  
            d_cdf[bi] += d_cdf[ai];
        }
    }

    if(idx == 0){
        d_cdf[255 + CONFLICT_FREE_OFFSET(255)] = 0;
    }

    #pragma omp unroll
    for(i = 128; i >= 1; i /= 2){
        __syncthreads();
        if((idx + 1)  % i == 0){
            int ai = 2 * idx + 1 - i;
            ai += CONFLICT_FREE_OFFSET(ai);
            int bi = 2 * idx + 1;
            bi += CONFLICT_FREE_OFFSET(bi);  
            float t = d_cdf[ai];
            d_cdf[ai] = d_cdf[bi];
            d_cdf[bi] += t;
        }
    }
    
    __syncthreads();
    int res1 = (int)(((float)d_cdf[idx + bankOffsetA] - min) * 255 / d + 0.5);
    int res2 = (int)(((float)d_cdf[idx2 + bankOffsetB] - min) * 255 / d + 0.5);
    if(res1 < 0)
        res1 = 0;
    else if(res1 > 255)
        res1 = 255;

    if(res2 < 0)
        res2 = 0;
    else if(res2 > 255)
        res2 = 255; 

    lut[idx] = res1;
    lut[idx2] = res2;
}

__global__ void histogram_eq(unsigned char * d_result, int width, unsigned char * d_img){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    d_result[y * width + x] = (unsigned char) c_lut[d_img[y * width + x]];    
}

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    //histogram
    int i;
    int nbr_bin = 256;
    int img_size = img_in.h * img_in.w;
    int *d_hist;
    unsigned char *d_img;
    CUDA_CHECK_ERROR( cudaMalloc(&d_hist, 256 * sizeof(int)) );
    CUDA_CHECK_ERROR( cudaMalloc(&d_img, sizeof(unsigned char) * img_in.w * img_in.h) );
    CUDA_CHECK_ERROR( cudaMemcpy(d_img, img_in.img, sizeof(unsigned char) * img_in.w * img_in.h, cudaMemcpyHostToDevice) );
    int *d_lut;
    CUDA_CHECK_ERROR( cudaMalloc(&d_lut, 256 * sizeof(int)) );
    unsigned char *d_result;
    CUDA_CHECK_ERROR( cudaMalloc(&d_result, img_in.w * img_in.h * sizeof(unsigned char)));

    #pragma omp unroll
    for ( i = 0; i < nbr_bin; i ++){   
        hist[i] = 0;
    }

    //histogram(hist, img_in.img, img_in.h * img_in.w, 256, img_in.w, img_in.h);
    //CUDA_CHECK_ERROR( cudaMemcpyToSymbol(c_hist, hist, 256 * sizeof(int)) );

    CUDA_CHECK_ERROR( cudaMemcpy(d_hist, hist, 256 * sizeof(int), cudaMemcpyHostToDevice) );
    
    dim3 dim_grid, dim_block;
    dim_block.x = 256;
    dim_block.y = 4;
    if(img_size / 1024 != 0){
        dim_grid.x = (img_in.w / 256) + 1;
        dim_grid.y = (img_in.h / 4);
    }
    else{
        dim_grid.x = (img_in.w / 256);
        dim_grid.y = (img_in.h / 4);
    }
    
    histogram <<< dim_grid, dim_block >>> (img_in.w, img_in.h, d_hist, d_img);
    CUDA_CHECK_ERROR( cudaMemcpyToSymbol(c_hist, d_hist, 256 * sizeof(int)) );
    
    cudaFree(d_hist);

    //histogram_equalization
    
    dim3 extra_block, extra_grid;
    extra_block.x = 128;
    extra_block.y = 1;
    extra_grid.x = 1;
    extra_grid.y = 1;

    prefix_sum <<< extra_grid, extra_block >>> (d_lut, img_size);

    CUDA_CHECK_ERROR( cudaMemcpyToSymbol(c_lut, d_lut, 256 * sizeof(int)) );
    cudaFree(d_lut);

    /* Get the result image */
    
    histogram_eq <<< dim_grid, dim_block >>> (d_result, img_in.w, d_img);

    CUDA_CHECK_ERROR( cudaMemcpy(result.img, d_result, img_in.w * img_in.h * sizeof(unsigned char), cudaMemcpyDeviceToHost) );

    cudaFree(d_result);
    cudaFree(d_img);

    return result;
}
