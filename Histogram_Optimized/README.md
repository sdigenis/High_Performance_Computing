# Histogram in CUDA

This project is about a GPU performed Histogram algorithm that was made and tested against a CPU one.<br>

## Histogram
A histogram is an approximate representation of the distribution of numerical data.<br>
For more information you can see the wikipedia page of [histogram](https://en.wikipedia.org/wiki/Histogram)

## How To Run

Firstly you should compile using the Makefile.
```bash
make 
```
Be aware because it's a CUDA application you need the nvcc compiler installed in your system.<br>
After compilation you just need to run it providing a .pgm file (file in black and white for better differences) and the name of the file you want to save the output picture.<br>
Samples of .pgm pictures you can find on the folder Images
<br>
eg.
```bash
./main path_to_the_picture.pgm path_to_the_output_picture.pgm
```

## Maximizing Performance 
In order to maximize the performance of the application several methods were used:
* Constant memory to store the histogram array which is constantly used as well as the array where the output is being calculated and saved as there are lots of calculations happening before saving them to the final output.
*  Algorithmically wise a better performace came after using the prefix-sum technique.
*  Constantly profilling (NVIDIA nvprof and Nsight Systems) to check if any configurations were having better or worse performance 

## System
The system we performed and maximized our application was using a GTX690 GPU Kepler architecture. More information about the GPU you can find on the deviceQuery.txt file.

## Disclaimer 
This application was made to practise the CUDA programming language and trying to engineer the best possible performance on a non-real problem. Though it was very helpfull for someone who wants to understand CUDA language as well how to use all of his GPU capabilities to the maximum.