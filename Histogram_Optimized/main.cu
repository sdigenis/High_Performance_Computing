#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <time.h>

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename);

double total_time = 0;

int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;

	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
    struct timespec  tv1, tv2;

    printf("Running contrast enhancement for gray-scale images.\n");
    /// measure total time for reading data
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    ///
    img_ibuf_g = read_pgm(argv[1]);
    ///
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    printf ("Total reading_file time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
      (double) (tv2.tv_sec - tv1.tv_sec));
    ///
    total_time += (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 + (double) (tv2.tv_sec - tv1.tv_sec);
    
    run_cpu_gray_test(img_ibuf_g, argv[2]);
    free_pgm(img_ibuf_g);
    cudaDeviceReset();
    
    printf ("Total time = %10g seconds\n", total_time);

	return 0;
}



void run_cpu_gray_test(PGM_IMG img_in, char *out_filename)
{
    //unsigned int timer = 0;
    PGM_IMG img_obuf;
    struct timespec  tv1, tv2;
    
    printf("Starting CPU processing...\n");

    /// measure total time for running the histogram
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    ///

    img_obuf = contrast_enhancement_g(img_in);

    ///
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    printf ("Total cpu_process time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
      (double) (tv2.tv_sec - tv1.tv_sec));
    ///
    total_time += (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 + (double) (tv2.tv_sec - tv1.tv_sec);
    ///
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    ///

    write_pgm(img_obuf, out_filename);

    ///
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    printf ("Total writing_to_file time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
      (double) (tv2.tv_sec - tv1.tv_sec));
    ///
    total_time += (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 + (double) (tv2.tv_sec - tv1.tv_sec);
    free_pgm(img_obuf);
}


PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    //this maybe could be done otherwise
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);   //this also could be propably be done faster
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}

