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
		//for(int j=0;j<128;j++){printf("%f\n",d_op[j]);}
		int index1,index2;
		int mask;
		int i= blockDim.x * blockIdx.x + threadIdx.x;
		mask = 1<<qubit;
	    index1 = i;
		index2 = mask ^i;
		if(((i >>  qubit) & 1) == 0)
		{
			
			__shared__ float s1[2];
			__shared__ float s2[2];
			
			for(int j=0;j<3;j=j+2){
				s1[threadIdx.x]=d_u[j]*d_ip[i];
			}
		__syncthreads();
			for(int k=1;k<4;k=k+2)
			{
				s2[threadIdx.x]=d_u[k]*d_ip[i+(1<< qubit)];
			}
			__syncthreads();
			for(int q=0;q<1;q++)
			{
			d_op[i]=s1[q]+s2[q];
			d_op[i+(1<< qubit)] =s1[q+1]+s2[q+1];	
			}
			
			
			//d_op[i] = (d_u[0] * d_ip[i]) + (d_u[1] * d_ip[i+(1<< qubit)]);
			//d_op[i+(1<< qubit)] = (d_u[2] * d_ip[i]) + (d_u[3] * d_ip[i+(1<< qubit)]);
			//printf("%f\n",d_ip[i]);
			//printf("%f\n",d_ip[i+(1<<0)]);
			

		}
}



int main(int argc, char *argv[])
{
    //qubit_circuit = argv[0]; //qubit circuit
	//printf("%d\n",qubit_circuit);
    //scanf("%d",&qubit_circuit);
    //scanf("%d",&qubit_oper);
    //qubit_oper      = argv[1]; //qubit operation
    input_file = argv[1]; // "input_for_qc7_q0_q2_q3_q4_q5_q6.txt";
    //input_1=argv[0];
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
    if(p>23)
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
    	cudaMalloc((void**)&d_ip,(count-6)*sizeof(float));
    	cudaMalloc((void**)&d_op,(count-6)*sizeof(float));
	
	int block_size = 256;
	 int grid_size = int(count/block_size);
	//dim3 grid(grid_size,grid_size);
	//dim3 threads(block_size, block_size);
	
	
	vector_array=(float**) malloc(sizeof(float*)*count-1);
	ip= (float*) malloc(sizeof(float)*count-6);
	op=(float*) malloc(sizeof(float)*count-6);
	float **col1;
	float *u1,*u2,*u3,*u4,*u5,*u6;
	int *qubit;
	col1=(float**)malloc(sizeof(float*)*2);
	u1=(float*)malloc(sizeof(float)*4);
	u2=(float*)malloc(sizeof(float)*4);
	u3=(float*)malloc(sizeof(float)*4);
	u4=(float*)malloc(sizeof(float)*4);
	u5=(float*)malloc(sizeof(float)*4);
	u6=(float*)malloc(sizeof(float)*4);
	qubit=(int*)malloc(sizeof(int)*6);
	int a=0;
	int b=0;
	int c=0;
	int d=0;
	int e=0;
	int f=0;
	int g=0;
	
	for(i=0;i<2;i++){
        col1[i]=(float*) malloc(sizeof(float)*2);
	}
    //printf("%d,",size);
    int test_size=count-6;
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
		u1[i]=num1;
		//printf("%f",col2[i]);
		//printf("\n");
		}
		if(i>3 && i<8){
		u2[a]=num1;
		a++;
		}
		if(i>7 && i<12){
		u3[b]=num1;
		b++;
		}
		if(i>11 && i<16){
		u4[c]=num1;
		c++;
		}
		if(i>15 && i<20){
		u5[d]=num1;
		d++;
		}
		if(i>19 && i<24){
		u6[e]=num1;
		e++;
		}
		//printf("%f",col1[i][c]);
		//printf("\n");
		for(int a=0;a<2;a++){
		    for(int b=0;b<2;b++){
		        col1[a][b]=col2[(a*2)+b];		   }
		}
		

        

		if(i>23 && i<(p-6)){

		    vector_array[l][0]=num1;
		    ip[l]=num1;
		    l++;
		    
		}

        i++;
        if(i>p-6){
            qubit_oper =  num1;
		qubit[g]=num1;
		g++;
            

		}
	}
	
	int num_frag = (count-6)/32;
	float *inp;
	float *onp;
	int k=0;
	int n=0;
	dim3 grid(2,256);
	
	for(i=0;i<num_frag;i++)
	{
		for(int j=0;j<32;j++)
		{
			inp[j]=ip[k];
			k++;
		}
	cudaMemcpy(d_u,u1,4*sizeof(float),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_ip,inp,32*sizeof(float),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_op,onp,32*sizeof(float),cudaMemcpyHostToDevice);
		
		mat_mul<<<grid, 32>>>(d_u,d_ip,d_op,qubit[0]);
	cudaMemcpy(onp,d_op,32*sizeof(float),cudaMemcpyDeviceToHost);
		for(int h=0;h<32;h++)
		{
			op[n]=onp[h]; 
			n++;
		}
	}
	
	for(i=0;i<count-6;i++)
	{
		ip[i]=op[i];
	}
	
	
	/*dim3 grid(2,256);
	
	 cudaMemcpy(d_u,u1,4*sizeof(float),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_ip,ip,(count-6)*sizeof(float),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_op,op,(count-6)*sizeof(float),cudaMemcpyHostToDevice);
	 
	 
	gettimeofday (&begin, NULL);
          
	mat_mul<<<grid, 256>>>(d_u,d_ip,d_op,qubit[0]);
    gettimeofday (&end, NULL);
	
	cudaMemcpy(ip,d_op,(count-6)*sizeof(float),cudaMemcpyDeviceToHost);
	//for(int j=0;j<count-6;j++){printf("%.3f\n",ip[j]);    }
	cudaFree(d_u);
	cudaFree(d_op);
	cudaFree(d_ip);
	
	cudaMemcpy(d_u,u2,4*sizeof(float),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_ip,ip,(count-6)*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_op,op,(count-6)*sizeof(float),cudaMemcpyHostToDevice);
	
	mat_mul<<<grid, 256>>>(d_u,d_ip,d_op,qubit[1]);
	
	cudaMemcpy(ip,d_op,(count-6)*sizeof(float),cudaMemcpyDeviceToHost);
	
	cudaMemcpy(d_u,u3,4*sizeof(float),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_ip,ip,(count-6)*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_op,op,(count-6)*sizeof(float),cudaMemcpyHostToDevice);
	
	mat_mul<<<grid, 256>>>(d_u,d_ip,d_op,qubit[2]);
	
	cudaMemcpy(ip,d_op,(count-6)*sizeof(float),cudaMemcpyDeviceToHost);
	
	cudaMemcpy(d_u,u4,4*sizeof(float),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_ip,ip,(count-6)*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_op,op,(count-6)*sizeof(float),cudaMemcpyHostToDevice);
	
	mat_mul<<<grid, 256>>>(d_u,d_ip,d_op,qubit[3]);
	
	cudaMemcpy(ip,d_op,(count-6)*sizeof(float),cudaMemcpyDeviceToHost);
	
	cudaMemcpy(d_u,u5,4*sizeof(float),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_ip,ip,(count-6)*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_op,op,(count-6)*sizeof(float),cudaMemcpyHostToDevice);
	
	mat_mul<<<grid, 256>>>(d_u,d_ip,d_op,qubit[4]);
	
	cudaMemcpy(ip,d_op,(count-6)*sizeof(float),cudaMemcpyDeviceToHost);
	
	
	cudaMemcpy(d_u,u6,4*sizeof(float),cudaMemcpyHostToDevice);
	 cudaMemcpy(d_ip,ip,(count-6)*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_op,op,(count-6)*sizeof(float),cudaMemcpyHostToDevice);
	
	mat_mul<<<grid, 256>>>(d_u,d_ip,d_op,qubit[5]);
	
	cudaMemcpy(op,d_op,(count-1)*sizeof(float),cudaMemcpyDeviceToHost);*/
	
	//mat_mul1(u,ip,op,count-1,qubit_oper);
	for(int j=0;j<count-6;j++){printf("%.3f\n",op[j]);    }
    fclose(FP);
}

