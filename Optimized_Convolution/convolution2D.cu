/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	 0.1
#define MAXCAP 67108864
//#define CPU

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
                                   
  extern __shared__ mine_t s_data[];

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

  int  position;
  mine_t cur_threshold, max_threshold, max = 0;

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
  //printf("********\nsize: %ld\n********\n", dimW * dimH * sizeof(mine_t));
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
  #ifdef CPU
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
  #endif

  mine_t *outputGPU;
  if(imageH <= 8192){
    cudaHostAlloc((void**)&outputGPU, dimH * dimW * sizeof(mine_t), cudaHostAllocDefault);
  }
  
  mine_t *Input;
  //cudaHostAlloc((void**)&Input, dimH * dimW * sizeof(mine_t), cudaHostAllocDefault);

  mine_t *f_output = (mine_t *)malloc(imageH * imageW * sizeof(mine_t));
  if(f_output == NULL){ 
    printf("Erorr in allocating memory\n");
    return -1;
  }
  cudaError_t error;


  //allocate memory for gpu
  

  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // something's gone wrong
    // print out the CUDA error as a string
    //cudaFree(d_OutputGPU);
    //cudaFree(d_Buffer);
    //cudaFree(d_Filter);
    //cudaFree(d_Input);
    cudaFreeHost(Input);
    cudaFreeHost(outputGPU);
    #ifdef CPU
    free(h_OutputCPU);
    free(h_Buffer);
    #endif
    free(h_Input);
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

  int times = imageH / 8192;

  mine_t *input_big[times * times];
  mine_t *output_big[times * times];

  if(imageW > 8192){
    for(i = 0; i < times*times; i++){
      cudaHostAlloc((void**)&input_big[i], dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t), \
      cudaHostAllocDefault);
      //input_big[i] = (mine_t*)malloc( dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t)); 
      cudaHostAlloc((void**)&output_big[i], dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t),\
      cudaHostAllocDefault);
      //output_big[i] = (mine_t*)malloc(dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t));
      memset(input_big[i], 0.0, dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t));
    }
  }
  //printf("here\n");
  if(imageW <= 8192){  
    cudaHostAlloc((void**)&Input, dimH * dimW * sizeof(mine_t), cudaHostAllocDefault);
    memset(Input, 0.0, dimH * dimW * sizeof(mine_t));
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
  }
  else{
    for(int h = 0; h < times * times; h++){
     for(i = 0; i < (MAXCAP/ imageH); i++){
         for(j = 0; j < imageW; j++){
             h_Input[h * ((MAXCAP/ imageH) * imageW) + i * imageW + j] = \
             input_big[h][filter_radius * dimW + filter_radius + i * dimW + j];
         }
     }
   }
  }
  #ifdef CPU
  // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
  printf("CPU computation...\n");
  clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
  convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
  convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
  clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

  printf ("Total time = %10g seconds\n",
			(mine_t) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
      (mine_t) (tv2.tv_sec - tv1.tv_sec));
      
  #endif
  // do GPU stuff

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  //printf("here\n");
  //cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(mine_t), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_Filter, h_Filter, FILTER_LENGTH * sizeof(mine_t));
  
  //invokate kernel
  dim3 dim_grid;
  dim3 dim_block;
  int block_size; 

  
  
  if(imageH <= 8192){
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
    cudaMalloc(&d_Input, dimH * dimW * sizeof(mine_t));
    cudaMalloc(&d_Buffer, dimH * dimW * sizeof(mine_t));
    cudaMalloc(&d_OutputGPU, dimH * dimW * sizeof(mine_t));
    cudaMemset(d_Buffer, 0.0, dimH * dimW * sizeof(mine_t));
    cudaMemset(d_OutputGPU, 0.0, dimH * dimW * sizeof(mine_t));
    cudaMemcpy(d_Input, Input, dimW * dimH * sizeof(mine_t), cudaMemcpyHostToDevice);
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

      #ifdef CPU
      free(h_OutputCPU);
      free(h_Buffer);
      #endif
      free(h_Input);
      printf("CUDA Error: %s\n", cudaGetErrorString(error));
      cudaDeviceReset();
      // we can't recover from the error -- exit the program
      return 1;
    }
    printf("GPU computation...\n");
    block_size = dim_block.x;
    int shared_square_row = (block_size) * (block_size);
    int shared_square_col = (block_size) * (block_size);
    
    convolutionRowGPU <<< dim_grid, dim_block, shared_square_row * sizeof(mine_t) >>> (d_Buffer, d_Input, block_size,/* d_Filter, */ \
                                                imageW, imageH, filter_radius);
    convolutionColumnGPU <<< dim_grid, dim_block, shared_square_col * sizeof(mine_t) >>> (d_OutputGPU, d_Buffer, block_size, /* d_Filter, */ \
                                                    imageW, imageH, filter_radius);
    
    cudaMemcpy(outputGPU, d_OutputGPU, dimW * dimH * sizeof(mine_t), cudaMemcpyDeviceToHost);

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
      //free host memory
      #ifdef CPU
      free(h_OutputCPU);
      free(h_Buffer);
      #endif
      free(h_Input);
      printf("CUDA Error: %s\n", cudaGetErrorString(error));
      cudaDeviceReset();
      // we can't recover from the error -- exit the program
      return 1;
    }
  }

  else{
    
    cudaStream_t stream[times * times];

    for(i = 0; i < times * times; i++){
      cudaStreamCreate(&stream[i]);
    }

    dim_block.x = 32; 
    dim_block.y = 32;
    dim_grid.x = imageW / 32;
    dim_grid.y = MAXCAP / (imageW * 32);
    block_size = dim_block.x;
    
    cudaMalloc(&d_Input,  dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t));
    cudaMalloc(&d_Buffer, dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t));
    cudaMalloc(&d_OutputGPU, dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t));
    cudaMemset(d_Input, 0.0, dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t));
    cudaMemset(d_Buffer, 0.0, dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t));
    cudaMemset(d_OutputGPU, 0.0, dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t));
    int shared_square_row = (block_size) * (block_size);
    int shared_square_col = (block_size) * (block_size);
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

      for(int v = 0; v < times * times; v++){
        free(input_big[i]);
        free(output_big[i]);
      }

      #ifdef CPU
      free(h_OutputCPU);
      free(h_Buffer);
      #endif
      free(h_Input);
      printf("CUDA Error: %s\n", cudaGetErrorString(error));
      cudaDeviceReset();
      // we can't recover from the error -- exit the program
      return 1;
    }
    k = 0;
    printf("GPU computation...\n");
    //for(i = 0; i < times; i = i +1){
        /* if(i % 2 == 0){
          k = 0;
        }
        else {
          k = 1;
        } */
        for(i = 0; i < times * times; i++){
          cudaMemcpyAsync(d_Input, input_big[i], dimW * ((MAXCAP/ imageH) + 2 * filter_radius)* sizeof(mine_t), cudaMemcpyHostToDevice, stream[i]);
        //cudaMemcpyAsync(d_Input, input_big[i], dimW * ((MAXCAP/ imageH) + 2 * filter_radius)* sizeof(mine_t), cudaMemcpyHostToDevice, stream[k+1]);
        }
        for(i = 0; i < times * times; i++){
          convolutionRowGPU <<< dim_grid, dim_block, shared_square_row * sizeof(mine_t) , stream[i] >>> (d_Buffer, d_Input, block_size,/* d_Filter, */ \
                                                    imageW, MAXCAP /imageH , filter_radius);
        //cudaDeviceSynchronize();
        //convolutionRowGPU <<< dim_grid, dim_block, shared_square_row * sizeof(mine_t) , stream[k+1] >>> (d_Buffer, d_Input, block_size,/* d_Filter, */ \
          imageW, MAXCAP /imageH , filter_radius);
        }
        //convolutionRowGPU <<< dim_grid, dim_block, shared_square_row * sizeof(mine_t) , stream1 >>> (d_Buffer, d_Input, block_size,/* d_Filter, */ \
          imageW, MAXCAP /imageH , filter_radius);
        for(i = 0; i < times * times; i++){
          convolutionColumnGPU <<< dim_grid, dim_block, shared_square_col * sizeof(mine_t) , stream[i]>>> (d_OutputGPU, d_Buffer, block_size, /* d_Filter, */ \
                                                        imageW, MAXCAP /imageH, filter_radius);

        //convolutionColumnGPU <<< dim_grid, dim_block, shared_square_col * sizeof(mine_t) , stream1>>> (d_OutputGPU, d_Buffer, block_size, /* d_Filter, */ \
                                                          imageW, MAXCAP /imageH, filter_radius);
        //convolutionColumnGPU <<< dim_grid, dim_block, shared_square_col * sizeof(mine_t) , stream[k+1]>>> (d_OutputGPU, d_Buffer, block_size, /* d_Filter, */ \
                                                            imageW, MAXCAP /imageH, filter_radius);
        }
        for(i = 0; i < times * times; i++){
          cudaMemcpyAsync(output_big[i], d_OutputGPU, dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t), cudaMemcpyDeviceToHost, stream[i]);
        //cudaMemcpyAsync(output_big[i], d_OutputGPU, dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t), cudaMemcpyDeviceToHost, stream[k+1]);
        }
        //cudaMemcpyAsync(output_big[i], d_OutputGPU, dimW * ((MAXCAP/ imageH) + 2 * filter_radius) * sizeof(mine_t), cudaMemcpyDeviceToHost, stream1);
        //printf("i: %d\n", i);
        /* error = cudaGetLastError();
        if(error != cudaSuccess)
        {
          // something's gone wrong
          // print out the CUDA error as a string
          cudaFree(d_OutputGPU);
          cudaFree(d_Buffer);
          //cudaFree(d_Filter);
          for(int v = 0; v < times * times; v++){
            cudaFreeHost(input_big[v]);
            cudaFreeHost(output_big[v]);
          }

          
          #ifdef CPU
          free(h_OutputCPU);
          free(h_Buffer);
          #endif
          free(h_Input);

          cudaFree(d_Input);
          printf("CUDA Error: %s\n", cudaGetErrorString(error));
          cudaFreeHost(Input);
          cudaFreeHost(outputGPU);
          cudaDeviceReset();
          // we can't recover from the error -- exit the program
          return 1;
        } */
        //cudaDeviceSynchronize();
      
    //}
  }
  //printf("here\n");
  //cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time, start, stop);
  printf ("Total time = %10g seconds\n",
			    (mine_t) gpu_time/1000);
  //end of gpu computation

  error = cudaGetLastError();
  if(imageW <= 8192){
    if(error != cudaSuccess)
    {
      // something's gone wrong
      // print out the CUDA error as a string
      cudaFree(d_OutputGPU);
      cudaFree(d_Buffer);
      //cudaFree(d_Filter);
      cudaFree(d_Input);


      free(h_OutputCPU);
      #ifdef CPU
      free(h_Buffer);
      #endif
      free(h_Input);
      printf("CUDA Error: %s\n", cudaGetErrorString(error));
      cudaFreeHost(Input);
      cudaFreeHost(outputGPU);
      cudaDeviceReset();
      // we can't recover from the error -- exit the program
      return 1;
    }
  }
  else{
    if(error != cudaSuccess)
    {
      // something's gone wrong
      // print out the CUDA error as a string
      cudaFree(d_OutputGPU);
      cudaFree(d_Buffer);
      //cudaFree(d_Filter);
      
      for(int v = 0; v < times * times; v++){
        cudaFreeHost(input_big[v]);
        cudaFreeHost(output_big[v]);
      }

      free(h_OutputCPU);
      #ifdef CPU
      free(h_Buffer);
      #endif
      free(h_Input);

      cudaFree(d_Input);
      printf("CUDA Error: %s\n", cudaGetErrorString(error));
      //cudaFreeHost(Input);
      //cudaFreeHost(outputGPU);
      cudaDeviceReset();
      // we can't recover from the error -- exit the program
      return 1;
    }
    for(i = 0; i < times * times; i++){
      cudaFreeHost(input_big[i]);
    }
  }
  
  
  
  if(imageH <= 8192){
    for (i = 0, c = 0; i < dimH; i++){
      if(i < filter_radius || i > imageH + filter_radius - 1 ){
        continue;
      }
      for(j = 0, k = 0; j < dimW; j++){
        if(j < filter_radius || j > imageW + filter_radius -1){
          continue;
        }
        f_output[c * imageW + k] = outputGPU[i * dimW + j];
        //printf("\t%.1f ", f_output[c * imageW + k]);
        k++;
      }
      c++;
      //printf("\n");
    }
  }
  else{
    int p = 0;
    for(i = 0; i < times * times ; i++){
        for(int h = 0, c = 0; h < (MAXCAP/ imageH) + 2 * filter_radius; h++){
            if(h < filter_radius || h > imageH + filter_radius - 1){
                continue;
            }
            for(int l = 0, k = 0; l < dimW; l++){
                if(l < filter_radius || l > imageW + filter_radius -1){
                    continue;
                }
                f_output[i * (MAXCAP/ imageH) + c * imageW + k] = output_big[i][h * dimW + l];
                k++;
            }
            c++;
        }
    }
  }
  if(imageW > 8192){
    for(i = 0; i < times * times; i++){
      cudaFreeHost(output_big[i]);
    }
  }
  else{
    cudaFreeHost(Input);
    cudaFreeHost(outputGPU);
  }
  #ifdef CPU
  for(i = 0; i < imageH * imageW; i++){
    cur_threshold = ABS(f_output[i] - h_OutputCPU[i]);
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
  #endif
  //free all gpu allocated memory
  cudaFree(d_OutputGPU);
  cudaFree(d_Buffer);
  cudaFree(d_Filter);
  cudaFree(d_Input);


  // free all the allocated memory
  
  #ifdef CPU
  free(h_OutputCPU);
  free(h_Buffer);
  #endif
  free(h_Input);
  //free(h_Filter);

  //free(outputGPU);
  free(f_output);
  //free(Input);
  cudaFreeHost(h_Filter);
  //cudaFreeHost(Input);
  //cudaFreeHost(outputGPU);
  // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
  cudaDeviceReset();


  return 0;
}
