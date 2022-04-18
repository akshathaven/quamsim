//Part A
/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <math.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *U, float *O, int numElements,int q,int lognumElements)
{
	  __shared__ float s[128];

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int segment1 = i/q;
	int j= i+ (numElements/4);
	int segment2=j/q;
	int i_four=4*threadIdx.x;

	
	i=i+segment1*q;
	j=j+segment2*q;
	
	 s[i_four]=A[i]; 
	 s[i_four+1]=A[i+q];
	 s[i_four+2]=A[j]; 
	 s[i_four+3]=A[j+q]; 
	
	 
   	  __syncthreads();

	  O[j]=(U[0]*s[i_four+2]) + (U[1]*s[i_four+3]);                 //first matrix (first half)
	  O[j+q]=(U[2]*s[i_four+2])+(U[3]*s[i_four+3])  ;  
	
	 O[i]=(U[0]*s[i_four]) + (U[1]*s[i_four+1]);                 //first matrix (first half)
	  O[i+q]=(U[2]*s[i_four])+(U[3]*s[i_four+1])  ; 
	 


}


/**
 * Host main routine
 */
int
main(int argc, char* argv[])
{
    char *input_file;
	input_file= argv[1];
	// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
  
    int temp_numElements=100000000;
    int num_qgate_elements=4;
    size_t size = temp_numElements * sizeof(float);
    size_t size_gate = num_qgate_elements * sizeof(float);
    


	float *temp_array=(float*)malloc(size);
    float *h_U1 = (float *)malloc(size_gate);
	float *h_U2 = (float *)malloc(size_gate);
	float *h_U3 = (float *)malloc(size_gate);
	float *h_U4 = (float *)malloc(size_gate);
	float *h_U5 = (float *)malloc(size_gate);
	float *h_U6 = (float *)malloc(size_gate);

    
    FILE *FP;
    float temp1;
   
    FP = fopen(input_file, "r");
    fscanf(FP,"%f %f %f %f",&h_U1[0],&h_U1[1],&h_U1[2],&h_U1[3]);
	fscanf(FP,"%f %f %f %f",&h_U2[0],&h_U2[1],&h_U2[2],&h_U2[3]);
	fscanf(FP,"%f %f %f %f",&h_U3[0],&h_U3[1],&h_U3[2],&h_U3[3]);
	fscanf(FP,"%f %f %f %f",&h_U4[0],&h_U4[1],&h_U4[2],&h_U4[3]);
	fscanf(FP,"%f %f %f %f",&h_U5[0],&h_U5[1],&h_U5[2],&h_U5[3]);
	fscanf(FP,"%f %f %f %f",&h_U6[0],&h_U6[1],&h_U6[2],&h_U6[3]);


    int j=0;
    while(fscanf(FP,"%f ",&temp1)!= EOF)
    {

	        temp_array[j]=temp1;
		    j++;
    }

	int numElements=j-6;
    size = numElements * sizeof(float);
 
	   // Allocate the host input vector A
    float *h_A = (float *)malloc(size);
	int l;
    for(l=0;l<(j-6);l++)
	    {h_A[l]=temp_array[l];}
	
	
    int q1=temp_array[j-6];
    int qbit1=pow(2,q1);   //qbit
	
	
	int q2=temp_array[j-5];
    int qbit2=pow(2,q2);   //qbit

	
	int q3=temp_array[j-4];
    int qbit3=pow(2,q3);   //qbit

	
	int q4=temp_array[j-3];
    int qbit4=pow(2,q4);   //qbit

	
	int q5=temp_array[j-2];
    int qbit5=pow(2,q5);   //qbit
	
	
	int q6=temp_array[j-1];
    int qbit6=pow(2,q6);   //qbit


    free(temp_array);
   
	
    float *h_O=(float *)malloc(size);
	float *h_O1=(float *)malloc(size);
	float *h_O2=(float *)malloc(size);
	float *h_O3=(float *)malloc(size);
	float *h_O4=(float *)malloc(size);
	float *h_O5=(float *)malloc(size);

    // Verify that allocations succeeded
  if (h_A == NULL || h_U1 == NULL || h_O == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

   

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_U1 = NULL;
    err = cudaMalloc((void **)&d_U1, size_gate);
	
	float *d_U2 = NULL;
    err = cudaMalloc((void **)&d_U2, size_gate);
	
	 float *d_U3 = NULL;
    err = cudaMalloc((void **)&d_U3, size_gate);
	
	float *d_U4 = NULL;
    err = cudaMalloc((void **)&d_U4, size_gate);
	
	float *d_U5 = NULL;
    err = cudaMalloc((void **)&d_U5, size_gate);
	
	float *d_U6 = NULL;
    err = cudaMalloc((void **)&d_U6, size_gate);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector U (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_O = NULL;
    err = cudaMalloc((void **)&d_O, size);
	
    float *d_O1 = NULL;
    err = cudaMalloc((void **)&d_O1, size);
	
    float *d_O2 = NULL;
    err = cudaMalloc((void **)&d_O2, size);
	
    float *d_O3 = NULL;
    err = cudaMalloc((void **)&d_O3, size);
	
    float *d_O4 = NULL;
    err = cudaMalloc((void **)&d_O4, size);
	
    float *d_O5 = NULL;
    err = cudaMalloc((void **)&d_O5, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector O (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
   // printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_U1, h_U1, size_gate, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_U2, h_U2, size_gate, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_U3, h_U3, size_gate, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_U4, h_U4, size_gate, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_U5, h_U5, size_gate, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_U6, h_U6, size_gate, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector U from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
int lognumElements= (int)(ceil(log(numElements) / log(2)));
    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock=32;
    int blocksPerGrid=(numElements/128) ;
	//int blocksPerGrid=(numElements + threadsPerBlock - 1) / threadsPerBlock;
	
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_U1, d_O, numElements,qbit1,lognumElements);

	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_O, d_U2, d_O1, numElements,qbit2,lognumElements);
	
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_O1, d_U3, d_O2, numElements,qbit3,lognumElements);

	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_O2, d_U4, d_O3, numElements,qbit4,lognumElements);
	
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_O3, d_U5, d_O4, numElements,qbit5,lognumElements);

	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_O4, d_U6, d_O5, numElements,qbit6,lognumElements);

	
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_O , d_O,  size, cudaMemcpyDeviceToHost);
	err = cudaMemcpy(h_O1, d_O1, size, cudaMemcpyDeviceToHost);
	err = cudaMemcpy(h_O2, d_O2, size, cudaMemcpyDeviceToHost);
	err = cudaMemcpy(h_O3, d_O3, size, cudaMemcpyDeviceToHost);
	err = cudaMemcpy(h_O4, d_O4, size, cudaMemcpyDeviceToHost);
	err = cudaMemcpy(h_O5, d_O5, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   int h;
   //printf("\n the output file is:");
   
   for(h=0;h<j-6;h++)
   {
	  printf("%0.3f",h_O5[h]);
	 printf("\n");
   }	 

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_U1);
	err = cudaFree(d_U2);
	err = cudaFree(d_U3);
	err = cudaFree(d_U4);
	err = cudaFree(d_U5);
	err = cudaFree(d_U6);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector U (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_O);
	err = cudaFree(d_O1);
	err = cudaFree(d_O2);
	err = cudaFree(d_O3);
	err = cudaFree(d_O4);
	err = cudaFree(d_O5);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector O (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_U1);
	free(h_U2);
	free(h_U3);
	free(h_U4);
	free(h_U5);
	free(h_U6);
	
    free(h_O);
	free(h_O1);
	free(h_O2);
	free(h_O3);
	free(h_O4);
	free(h_O5);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   // printf("Done\n");
    return 0;
}

