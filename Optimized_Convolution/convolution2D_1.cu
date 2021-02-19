/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
 
unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	 0.1

#ifndef ACCYRACY
#define ACCYRACY
typedef double mine_t;
#else
typedef float mine_t;
#endif

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(mine_t *h_Dst, mine_t *h_Src, mine_t *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      mine_t sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;
        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }
      }
      h_Dst[y * imageW + x] = sum;
    }
  } 
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(mine_t *h_Dst, mine_t *h_Src, mine_t *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      mine_t sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }
      }
      h_Dst[y * imageW + x] = sum;
    }
  } 
}

/**************GPU Kernel***************/

__constant__ mine_t d_Filter[513];

__global__ void convolutionRowGPU(mine_t *d_Dst, mine_t *d_Src, int block_size,/* mine_t *d_Filter, */ \
                                  int imageW, int imageH, int filterR){
  int dimensionW = imageW + 2 * filterR;
  int dimensionH = imageH + 2 * filterR;
  int k , d;
  int x = threadIdx.x + blockDim.x * blockIdx.x + filterR;
  int y = threadIdx.y + blockDim.y * blockIdx.y + filterR;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  mine_t sum = 0.0;
  mine_t src, filter;
  int s_dim = block_size;
                                   
  __shared__ mine_t s_data[32*32];
  s_data[ty * s_dim + tx] = d_Src[y * dimensionW + x];

  __syncthreads();
  
  for(k = -filterR; k <= filterR; k++){
    d = k + tx;
    if(d >= 0 && d < s_dim){
      src = s_data[ty * s_dim + d];
    }
    else{
      src = d_Src[y * dimensionW + k + x];
    }
    filter = d_Filter[filterR - k]; 
    sum = sum + (src * filter);
  } 
  d_Dst[y * dimensionW + x] = sum;
}

__global__ void convolutionColumnGPU(mine_t *d_Dst, mine_t *d_Src, int block_size,/* mine_t *d_Filter, */\
                                        int imageW, int imageH, int filterR) {
  int dimensionW = imageW + 2 * filterR;
  int dimensionH = imageH + 2 * filterR;
  int k , d;
  int x = threadIdx.x + blockDim.x * blockIdx.x + filterR;
  int y = threadIdx.y + blockDim.y * blockIdx.y + filterR;
  mine_t sum = 0.0;
  mine_t src , filter;
  int s_dim = block_size;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  extern __shared__ mine_t s_data[];

  s_data[ty * s_dim + tx] = d_Src[y * dimensionW + x];

  __syncthreads();

  for(k = -filterR; k <= filterR; k++){
    d =  ty + k;
    if(d >= 0 && d < s_dim){
      src = s_data[d * s_dim + tx];
    }
    else{
      src = d_Src[(y+k) * dimensionW + x];
    }
    
    filter = d_Filter[filterR - k]; 
    sum = sum + (src * filter);
  }
  d_Dst[y * dimensionW + x] = sum;
}

/*****************************/


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
  mine_t
  *h_Filter, /* *d_Filter, */
  *h_Input, *d_Input,
  *h_Buffer, *d_Buffer,
  *h_OutputCPU, *d_OutputGPU;

  int imageW;
  int imageH;
  unsigned int i, j, k;

  // variables for time
  struct timespec  tv1, tv2;
  cudaEvent_t start, stop;
  float gpu_time = 0.0f;  


  printf("Enter filter radius : ");
  scanf("%d", &filter_radius);

  // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
  // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
  // Gia aplothta thewroume tetragwnikes eikones.  

  printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
  scanf("%d", &imageW);
  //imageW = 16384;
  imageH = imageW;
  int dimH = imageH + 2 * filter_radius;
  int dimW = imageW + 2 * filter_radius;
  printf("Accuracy: %f\n", accuracy);
  printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
  //printf("********\nsize: %ld\n********\n", dimW * dimH * sizeof(double));
  printf("Allocating and initializing host arrays...\n");
  // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
  /* h_Filter = (mine_t *)malloc(FILTER_LENGTH * sizeof(mine_t));
  if(h_Filter == NULL){
    printf("Erorr in allocating memory\n");
    return -1;
  } */
  cudaHostAlloc((void**)&h_Filter, FILTER_LENGTH * sizeof(mine_t), cudaHostAllocDefault);
  h_Input = (mine_t *)malloc(imageW * imageH * sizeof(mine_t));
  if(h_Input == NULL){ 
    printf("Erorr in allocating memory\n");
    return -1;
  }
  h_Buffer = (mine_t *)malloc(imageW * imageH * sizeof(mine_t));
  if(h_Buffer == NULL){
    printf("Erorr in allocating memory\n");
    return -1;
  }
  h_OutputCPU = (mine_t *)malloc(imageW * imageH * sizeof(mine_t));
  if(h_OutputCPU == NULL){
    printf("Erorr in allocating memory\n");
    return -1;
  }

  /* mine_t *outputGPU = (mine_t *)malloc(dimH * dimW * sizeof(mine_t));
  if(outputGPU == NULL){
    printf("Erorr in allocating memory\n");
    return -1;
  } */
  mine_t *outputGPU;
  cudaHostAlloc((void**)&outputGPU, dimH * dimW * sizeof(mine_t), cudaHostAllocMapped);

  /* mine_t *Input = (mine_t *)malloc(dimH * dimW * sizeof(mine_t));
  if(h_Input == NULL){ 
    printf("Erorr in allocating memory\n");
    return -1;
  } */
  mine_t *Input;
  cudaHostAlloc((void**)&Input, dimH * dimW * sizeof(mine_t), cudaHostAllocMapped);

  mine_t *f_output = (mine_t *)malloc(imageH * imageW * sizeof(mine_t));
  if(f_output == NULL){ 
    printf("Erorr in allocating memory\n");
    return -1;
  }
  cudaError_t error;

  memset(Input, 0.0, dimH * dimW * sizeof(mine_t));

  //allocate memory for gpu
  //cudaMalloc(&d_Filter, FILTER_LENGTH * sizeof(mine_t));
  cudaMalloc(&d_Input, dimH * dimW * sizeof(mine_t));
  cudaMalloc(&d_Buffer, dimH * dimW * sizeof(mine_t));
  cudaMalloc(&d_OutputGPU, dimH * dimW * sizeof(mine_t));

  cudaMemset(d_Buffer, 0.0, dimH * dimW * sizeof(mine_t));
  cudaMemset(d_OutputGPU, 0.0, dimH * dimW * sizeof(mine_t));

  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // something's gone wrong
    // print out the CUDA error as a string
    cudaFree(d_OutputGPU);
    cudaFree(d_Buffer);
    //cudaFree(d_Filter);
    cudaFree(d_Input);
    cudaFreeHost(Input);
    cudaFreeHost(outputGPU);
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
    cudaDeviceReset();
    // we can't recover from the error -- exit the program
    return 1;
  }
  
  // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
  // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
  // to convolution kai arxikopoieitai kai auth tuxaia.
  srand(200);
  for (i = 0; i < FILTER_LENGTH; i++) {
      h_Filter[i] = (mine_t)(rand() % 16);
  }

  for (i = 0; i < imageW * imageH; i++) {
    h_Input[i] = (mine_t)rand() / ((mine_t)RAND_MAX / 255) + (mine_t)rand() / (mine_t)RAND_MAX;
  }
  int c;
  
  for (i = 0, c = 0; i < dimH; i++){
    if(i < filter_radius || i > imageH + filter_radius - 1){
      for(j = 0; j < dimW; j++){
      }
      continue;
    }
    for(j = 0, k = 0; j < dimW; j++){
      if(j < filter_radius || j > imageW + filter_radius -1){
        continue;
      }
      Input[i * (dimW) + j] = h_Input[c * imageW + k];
      k++;
    }
    c++;
  }

  
  /* mine_t *filter = (mine_t *)malloc( sizeof(mine_t) * FILTER_LENGTH);
  memcpy(filter, h_Filter, FILTER_LENGTH * sizeof(mine_t)); */

  
  
  // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
  printf("CPU computation...\n");
  clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
  convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
  convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
  clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

  printf ("Total time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
      (double) (tv2.tv_sec - tv1.tv_sec));
      

  // do GPU stuff

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  //printf("here\n");
  //cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(mine_t), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_Filter, h_Filter, FILTER_LENGTH * sizeof(mine_t));
  cudaMemcpy(d_Input, Input, dimW * dimH * sizeof(mine_t), cudaMemcpyHostToDevice);
  
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // something's gone wrong
    // print out the CUDA error as a string
    cudaFree(d_OutputGPU);
    cudaFree(d_Buffer);
    //cudaFree(d_Filter);
    cudaFreeHost(Input);
    cudaFreeHost(outputGPU);
    cudaFree(d_Input);
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
    cudaDeviceReset();
    // we can't recover from the error -- exit the program
    return 1;
  }
  //invokate kernel
  dim3 dim_grid;
  dim3 dim_block;
  int block_size; 

  if(imageH <= 32){
    dim_grid.x = 1;
    dim_grid.y = 1;
    dim_block.x = imageW;
    dim_block.y = imageH;
  }
  else{
    dim_block.x = 32; 
    dim_block.y = 32;
    dim_grid.x = imageW/32;
    dim_grid.y = imageH/32;
  }
  block_size = dim_block.x;
  int shared_square_row = (block_size) * (block_size);
  int shared_square_col = (block_size) * (block_size + 1);
  printf("GPU computation...\n"); 
  convolutionRowGPU <<< dim_grid, dim_block, shared_square_row * sizeof(mine_t) >>> (d_Buffer, d_Input, block_size,/* d_Filter, */ \
                                              imageW, imageH, filter_radius);
  //cudaDeviceSynchronize();
  //cudafree(input);
  //cudamalloc(outpugpu)
  

  convolutionColumnGPU <<< dim_grid, dim_block, shared_square_col * sizeof(mine_t) >>> (d_OutputGPU, d_Buffer, block_size, /* d_Filter, */ \
                                                  imageW, imageH, filter_radius);
  //cudaDeviceSynchronize();
  cudaMemcpy(outputGPU, d_OutputGPU, dimW * dimH * sizeof(mine_t), cudaMemcpyDeviceToHost);
  //double *out;
  //cudaHostGetDevicePointer(&out, d_OutputGPU, 0);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // something's gone wrong
    // print out the CUDA error as a string
    cudaFree(d_OutputGPU);
    cudaFree(d_Buffer);
    //cudaFree(d_Filter);
    cudaFreeHost(Input);
    cudaFreeHost(outputGPU);
    cudaFree(d_Input);
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
    cudaDeviceReset();
    // we can't recover from the error -- exit the program
    return 1;
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time, start, stop);
  printf ("Total time = %10g seconds\n",
			    (double) gpu_time/1000);
  //end of gpu computation

  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // something's gone wrong
    // print out the CUDA error as a string
    cudaFree(d_OutputGPU);
    cudaFree(d_Buffer);
    //cudaFree(d_Filter);

    cudaFree(d_Input);
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
    cudaDeviceReset();
    // we can't recover from the error -- exit the program
    return 1;
  }
  

  for (i = 0, c = 0; i < dimH; i++){
    if(i < filter_radius || i > imageH + filter_radius - 1 ){
      continue;
    }
    for(j = 0, k = 0; j < dimW; j++){
      if(j < filter_radius || j > imageW + filter_radius -1){
        continue;
      }
      f_output[c * imageW + k] = outputGPU[i * dimW + j];
      k++;
    }
    c++;
  }
  
  // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
  // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
  int  position;
  mine_t cur_threshold, max_threshold, max = 0 ;
  for(i = 0; i < imageH * imageW; i++){
    cur_threshold = ABS(f_output[i] - h_OutputCPU[i]);
    //printf("%f  vs  %f\n", h_OutputCPU[i], outputGPU[i]);
    if(cur_threshold > accuracy){
      for(j = i; j < imageH * imageW; j++){
        max_threshold = ABS(f_output[j] - h_OutputCPU[j]);
        if(max < max_threshold){
          max = max_threshold;
          position = j;
        }
      }
      printf("Max threshold: %f, at position: %d\n", max, position);
      printf("i: %d = %f\n", i, cur_threshold);
      printf("Inaccurate GPU output!\nExiting...\n");
      break;
      }
  }

  //free all gpu allocated memory
  cudaFree(d_OutputGPU);
  cudaFree(d_Buffer);
  cudaFree(d_Filter);
  cudaFree(d_Input);


  // free all the allocated memory
  free(h_OutputCPU);
  free(h_Buffer);
  free(h_Input);
  //free(h_Filter);

  //free(outputGPU);
  free(f_output);
  //free(Input);
  //cudaFreeHost(h_Filter);
  cudaFreeHost(Input);
  cudaFreeHost(outputGPU);
  // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
  cudaDeviceReset();


  return 0;
}
