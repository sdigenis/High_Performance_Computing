# Parallel K-Means Data Clustering

## Disclaimer

THIS SOFTWARE IS NOT MINE. I HAVE USED THE CODE OF IT TO GET FAMILIAR WITH OpenMP AND VTune Profiller.<br>
THE ONLY FILE I HAVE MODIFIED IS THE kmeans.c ALL THE OTHER FILES HAVE BEEN PROVIDED BY NORTHWESTERN UNIVERSITY.<br>

## Running 
  In order to run this software the run.sh script has written to make use of intel's  icc compiler.<br>
  There you can make any changes in order to run the project differently.

## Performance
The software's performance was measured via intel's VTune profiler and optimized for a remoter system running two processors 16-way multicore each, 2-way SMT (hyperthreaded).

The software package of parallel K-means data clustering contains the 
a parallel version in C <br>

## To run:
  * The Makefile will produce the "seq_main" executable for 
    thesequential version

  * The list of available command-line arguments can be obtained by
    running -h option
     o For example, running command "omp_main -h" will produce:
       Usage: main [switches] -i filename -n num_clusters
             -i filename    : file containing data to be clustered
             -c centers     : file containing initial centers. default: filename
             -b             : input file is in binary format (default no)
             -n num_clusters: number of clusters (K must > 1)
             -t threshold   : threshold value (default 0.0010)
             -p nproc       : number of threads (default system allocated)
             -a             : perform atomic OpenMP pragma (default no)
             -o             : output timing results (default no)
             -d             : enable debug mode


  ## Example run commands:
      # sequential K-means ----------------------------------------------------
      ```bash
      seq_main -o -b -n 4 -i Image_data/color17695.bin
      seq_main -o -b -n 4 -i Image_data/edge17695.bin
      seq_main -o -b -n 4 -i Image_data/texture17695.bin

      seq_main -o    -n 4 -i Image_data/color100.txt
      seq_main -o    -n 4 -i Image_data/edge100.txt
      seq_main -o    -n 4 -i Image_data/texture100.txt
      ```


Input file format:
The executables read an input file that stores the data points to be 
clustered. A few example files are provided in the sub-directory 
./Image_data. The input files can be in two formats: ASCII text and raw 
binary.

  * ASCII text format:
    o Each line contains the ID and coordinates of a single data point
    o The number of coordinates must be equal for all data points
  * Raw binary format:
    o There is a file header of 2 integers:
      *  The first 4-byte integer must be the number of data points.
      *  The second integer must be the number of coordinates.
    o The rest of the file contains the coordinates of all data 
      points and each coordinate is of type 4-byte float.

Output files: There are two output files:
  * Coordinates of cluster centers
    o The file name is the input file name appended with ".cluster_centres".
    o File extensions will be added, eg. ".txt" for ASCII format, and ".bin" 
      for binary.
    o For ASCII, each line contains an integer indicating the cluster id and
      the coordinates of the cluster center.
  * Membership of all data points to the clusters
    o The file name is the input file name appended with ".membership".
    o File extensions will be added, eg. ".txt" for ASCII format, and ".bin" 
      for binary.
    o For ASCII, each line contains two integers: data point index (from 0 to 
      the number of points) and the cluster id indicating the membership of
      the point.

Limitations:
    * Data type -- This implementation uses C float data type for all
      coordinates and other real numbers.
    * Large number of data points -- The number of data points cannot
      exceed 2G due to the 4-byte integers used in the programs.

