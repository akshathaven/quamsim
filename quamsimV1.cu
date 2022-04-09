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
//float col1[2][2];
//float *col1[2];
float col2[4];
int index1,index2;
int qubit_circuit;
int qubit_oper;
int mask;
//float (*mat1)[2];
float (*mat2)[1];
float (*res)[1];
struct timeval begin, end;

void mat_mul(float **col1,float **vector_array, int size,int qubit_oper)
{
    //printf("why");

    int arr_bool[size];
	for(int i=0;i<size;i++){
		arr_bool[i]=0;
	}

	for(int j=0;j<size-1;j++)
	{
		float res_array[2][1];
		memset(res_array, 0, sizeof(res_array));
		if (arr_bool[j]==1){
			//printf("%d",i);
		}
		else
		{
			//float **vec_mat;
			float col2[2][1];
			mask = 1<<qubit_oper;
			index1 = j;
			index2 = mask ^j;
			col2[0][0]=vector_array[index1][0];
			col2[1][0]=vector_array[index2][0];
			mat2=col2;
			arr_bool[index1]=1;
			arr_bool[index2]=1;
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 1; ++j) {
                    for (int k = 0; k < 2; ++k) {
                    res_array[i][j] += col1[i][k] * mat2[k][j];
                }
      }
  }
  res=res_array;
    vector_array[index1][0]=res[0][0];
    vector_array[index2][0]=res[1][0];

		}
	}

	for (int i = 0; i < size; ++i) {
      for (int j = 0; j < 1; ++j) {
         printf("%.3f  ", vector_array[i][j]);
         printf("\n");
      }
  }
}
void mat_mul1(float* u,float* ip,float *op,int size,int qubit_oper){
    printf("%d\n",qubit_oper);
	
    for(int j=0;j<size-1;j++){
       
        mask = 1<<qubit_oper;
	    index1 = j;
		index2 = mask ^j;
		
		if(((j>>qubit_oper)&1)==0){
		op[index1]=(u[0]*ip[index1])+(u[1]*ip[index2]);
		op[index2]=(u[2]*ip[index1])+(u[3]*ip[index2]);}
		
    }
    for(int j=0;j<size-1;j++){printf("%.3f\n",op[j]);    }
    
}

__global__ void mat_mul(float *d_u, float *d_ip,float *d_op,int qubit)
{
		
		int index1,index2;
		int mask;
		int i= blockDim.x * blockIdx.x + threadIdx.x;
		mask = 1<<2;
	    index1 = i;
		index2 = mask ^i;
		if(((i >> 2) & 1) == 0)
		{
			d_op[index1] = (d_u[0] * d_ip[index1]) + (d_u[1] * d_ip[index2]);
			d_op[index2] = (d_u[2] * d_ip[index1]) + (d_u[3] * d_ip[index2]);

		}
}



int main(int argc, char *argv[])
{
    //qubit_circuit = argv[0]; //qubit circuit
    //scanf("%d",&qubit_circuit);
    //scanf("%d",&qubit_oper);
    //qubit_oper      = argv[1]; //qubit operation
    input_file = "input1.txt";
    FP = fopen(input_file, "r");
    if (FP == NULL)
    {
        printf("Unable to open file %s\n", input_file);
        return 1;
    }
	int coun=0;
    int p=0;



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
	//printf("%d,",p);
	//printf("%d",count);
	float **vector_array;
	float *ip;
	float *op;
	
	float *d_u,*d_ip,*d_op;
	int d_qopr;
	
	cudaMalloc((void**)&d_u,4*sizeof(float));
    cudaMalloc((void**)&d_ip,(count-1)*sizeof(float));
    cudaMalloc((void**)&d_op,(count-1)*sizeof(float));
	
	
	
	
	
	vector_array=(float**) malloc(sizeof(float*)*count-1);
	ip= (float*) malloc(sizeof(float)*count-1);
	op=(float*) malloc(sizeof(float)*count-1);
	float **col1;
	float *u;
	col1=(float**)malloc(sizeof(float*)*2);
	u=(float*)malloc(sizeof(float)*4);
	for(i=0;i<2;i++){
        col1[i]=(float*) malloc(sizeof(float)*2);
	}
    //printf("%d,",size);
    int test_size=count-1;
	for(i=0;i<count-1;i++){
        vector_array[i]=(float*) malloc(sizeof(float)*1);
        
	}
	op=(float*)malloc(sizeof(float)*test_size);
	ip=(float*)malloc(sizeof(float)*test_size);
	fseek(FP,0,SEEK_SET);
    //open trace file to read
	i=0;
	int l=0;


	while(fscanf(FP, "%f", &num1) != EOF)
    {
		if(i<4){
		col2[i]=num1;
		u[i]=num1;
		//printf("%f",col2[i]);
		//printf("\n");
		}
		//printf("%f",col1[i][c]);
		//printf("\n");
		for(int a=0;a<2;a++){
		    for(int b=0;b<2;b++){
		        col1[a][b]=col2[(a*2)+b];		    }
		}
		

        

		if(i>3 && i<(p-1)){

		    vector_array[l][0]=num1;
		    ip[l]=num1;
		    l++;
		    
		}

        i++;
        if(i==p){
            qubit_oper =  num1;
            

		}
	}
	
	
	dim3 grid(1,2);
	
	 cudaMemcpy(d_u,u,4*sizeof(float),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_ip,ip,(count-1)*sizeof(float),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_op,op,(count-1)*sizeof(float),cudaMemcpyHostToDevice);
	 
	 
	gettimeofday (&begin, NULL);
          
	mat_mul<<<grid,1>>>(d_u,d_ip,d_op,qubit_oper);
    gettimeofday (&end, NULL);
	cudaMemcpy(op,d_op,(count-1)*sizeof(float),cudaMemcpyDeviceToHost);
	
	//mat_mul1(u,ip,op,count-1,qubit_oper);
	for(int j=0;j<count-1;j++){printf("%.3f\n",op[j]);    }
    fclose(FP);
}

