# CUDA Convolution

This software was made to get to know CUDA and how to try and get the maximum performance out of a GPU.<br>
It performs a convolution to an image with two different ways:
- A CPU sequantial one 
- A GPU optimized one 
both ways outcomes are being compared at the end to test the correctness and accuracy of the GPU way.<br>

To maximize accuracy double varieables have been used.<br>

## System and Performance 
In order to maximize GPU performance constant memory, shared memory, streams and better memory allocation ways have been used.<br>
The performance was measured and optimized for a remote system running a Tesla K80 GPU. More information for the GPU can be found in the deviceQuery.txt . <br>

## How to run

In order to run the application you don't need to provide any parameters.<br>
```bash
./convolution2D
```
After the application start running it will ask for a user's input for the convolution filter radius and then a number that the application will get as image's width and it has to be a power of 2.<br>
All images are square, so their width equals their height. The images are basically a poor implamention of them as they are 2D arrays that have random numbers from 0 to 255 that are being provided in a for loop. <br>

At the end the application will print the time it needed for the CPU to run as well as the total time for the GPU to show how quicker a simpler application could run on a GPU.<br>

If the produced outcome would be different from the CPU convolution process and the GPU one more than an accuracy number pre-given then an error comes up printing how much difference there is between the points and the application exits.<br>

## Maximizing Performance

In order to maximize performance lots of configuration have happened to the starting code:
- Starting with profilling to check the where the application was needing most of the time, so this way the first configuration was to change the way the copies to the DRAM 
- After that the use of constant memory for the always stable filter was made helped a lot 
- Using the shared memory as well helped a lot as there was a 30% reduce in the time in the testing sample which was a 16384x16384 "picture". 
- Finally the addition of partly process made possible to get bigger images with no the DRAM size not a being a problem. Combined with streams helped the application run in almost maximum speeds and the only limitations was the systems RAM memory.   

