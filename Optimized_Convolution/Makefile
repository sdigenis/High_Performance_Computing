# Makefile

CC = nvcc
OPTFLAGS = -g -O4

convolution2: convolution2D_1.cu convolution2D_3.cu convolution2D_4.cu
	$(CC) $(OPTFLAGS) convolution2D_1.cu -o convolution2D_1
	$(CC) $(OPTFLAGS) convolution2D_3.cu -o convolution2D_3
	$(CC) $(OPTFLAGS) convolution2D_4.cu -o convolution2D_4

1:	convolution2D_1.cu
	$(CC) $(OPTFLAGS) convolution2D_1.cu -o convolution2D_1

3:	convolution2D_3.cu
	$(CC) $(OPTFLAGS) convolution2D_3.cu -o convolution2D_3

4:	convolution2D_4.cu
	$(CC) $(OPTFLAGS) convolution2D_4.cu -o convolution2D_4

clean:
	rm convolution2D_1	convolution2D_3	convolution2D_4 
