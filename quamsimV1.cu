#include <stdio.h>
//#include <conio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

FILE *FP;    // to store trace file
char *input_file;
int i;

float num1,num2;
float col1[2][2];
float col2[4];
int index1,index2;
int qubit_circuit;
int qubit_oper;
int mask;

 struct timeval begin, end;

  //kernel<<<grid, block>>>();
  
  
  
__global__ void mat_mul(float *a, float *b,float *c)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	 
	c[1*y+x]=0;
	for(int k=0;k<2;k++)
	{
		c[1*y+x]=c[1*y+x]+a[2*y+k]*b[1*k+x];
	}
	
}






int main(int argc, char *argv[])
{
    //qubit_circuit = argv[0]; //qubit circuit
    //scanf("%d",&qubit_circuit);
    //scanf("%d",&qubit_oper);
    //qubit_oper      = argv[1]; //qubit operation
    input_file = argv[1];//"input1.txt";
    FP = fopen(input_file, "r");
    if (FP == NULL)
    {
        printf("Unable to open file %s\n", input_file);
        return 1;
    }
	int coun=0;
    int p=0;
	float res[2][1];
	float *d_mat1,*d_mat2,*d_res;
    //struct timeval start_time,end_time,elapsed_time;


	cudaMalloc((void**)&d_mat1,2*2*sizeof(float));
    cudaMalloc((void**)&d_mat2,2*1*sizeof(float));
    cudaMalloc((void**)&d_res,2*1*sizeof(float));
	
	
	int mask;
	//int size = pow(2,qubit_circuit);

    int count=0;
    //int p=0;
	while(fscanf(FP, "%f", &num1) != EOF){
    if(p>3)
        {
        count++;}
    p++;
	}
	//printf("%d",count);
	float **vector_array;
	vector_array=(float**) malloc(sizeof(float*)*count-1);
    //printf("%d,",size);
	for(i=0;i<count-1;i++){
        vector_array[i]=(float*) malloc(sizeof(float)*1);
	}
	fseek(FP,0,SEEK_SET);
    //open trace file to read
	i=0;
	int l=0;


	while(fscanf(FP, "%f", &num1) != EOF)
    {
		if(i<4){
		col2[i]=num1;
		//printf("%f",col1[i][c]);
	//printf("\n");
		}
		//printf("%f",col1[i][c]);
		//printf("\n");
		for(int a=0;a<2;a++){
		    for(int b=0;b<2;b++){
		        col1[a][b]=col2[(a*2)+b];		    }
		}

		/*if(i>3)
        {

		    vector_array[l][0]=num1;
		    //printf("%f",vector_array[l][0]);
		    //printf("\n");
		    l++;
		    //printf("%d,",l);
		}*/
        
          if(i>3 && i<(p-1))
        {

                    vector_array[l][0]=num1;
                    //printf("%f",vector_array[l][0]);
                    //printf("\n");
                    l++;
                    //printf("%d,",l);
        }
        i++;
        if(i==p)
        {
            qubit_oper = num1;
        }
        
        //printf("hh");
	}
	int arr_bool[count-1];
	for(int i=0;i<count-1;i++){
		arr_bool[i]=0;
	}

	for(int j=0;j<count-1;j++)
	{
		float res_array[2][1];
		memset(res_array, 0, sizeof(res_array));
		if (arr_bool[j]==1){
			//printf("%d",i);
		}
		else{
			//float **vec_mat;
			float col2[2][1];
			mask = 1<<qubit_oper;
			index1 = j;
			index2 = mask ^j;
			col2[0][0]=vector_array[index1][0];
			col2[1][0]=vector_array[index2][0];
			arr_bool[index1]=1;
			arr_bool[index2]=1;
            cudaMemcpy(d_mat1,col1,2*2*sizeof(float),cudaMemcpyHostToDevice);
			cudaMemcpy(d_mat2,col2,2*1*sizeof(float),cudaMemcpyHostToDevice);
			dim3 grid(1,2);
			  gettimeofday (&begin, NULL);
            
			mat_mul<<<grid,1>>>(d_mat1,d_mat2,d_res);
             gettimeofday (&end, NULL);
             
             
           //timersub(&start_time,&end_time,&elapsed_time);
           
           
			cudaMemcpy(res,d_res,2*1*sizeof(float),cudaMemcpyDeviceToHost);
			
			vector_array[index1][0]=res[0][0];
			vector_array[index2][0]=res[1][0];

		}
	}
         int time_in_us = 1e6 * (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec);
          //  printf("Timing: %d",time_in_us);
          
	for (int i = 0; i < count-1; ++i) {
      for (int j = 0; j < 1; ++j) {
         printf("%.3f  ", vector_array[i][j]);
         printf("\n");
      }
  }
	
	
	//printf("hh");
	//mat_mul(col1,vector_array,count,qubit_oper);
    fclose(FP);
}



