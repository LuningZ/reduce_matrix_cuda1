/**
 * @file float.cu
 * @brief Parallel implementation (in the CPU and GPU) to reduce a matrix to a single value
 * 		Code for Task2 of MAP55616 Assignment1.
 * @author Luning
 * @version 1.0
 * @date 2023-03-17
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<stdbool.h>
#include<sys/time.h>

#define BLOCK_SIZE 256

float* init_matrix(int n, int m);
float* row_abs_sum(float* matrix, int n, int m);
float* col_abs_sum(float* matrix, int n, int m);
float vectorReduce(float* vec, int size);
double get_time(void);

/*
 * Kernel function to adds together the absolute value of each element of each row.
 * (into a vector of size n)
 */
__global__ void row_abs_sum_kernel(float* matrix, float* result, int n ,int m){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row<n){
		float sum = 0.0f;
		for(int j=0;j<m;j++){
			sum += fabs(matrix[row*m + j]);
		}
		result[row] = sum;
	}
}

/*
 * Kernel function to adds together the absolute value of each element of each column.
 * (into a vector of size m)
 */
__global__ void col_abs_sum_kernel(float* matrix, float *result, int n, int m){
	int col = blockIdx.x* blockDim.x + threadIdx.x;
	if(col<m){
		float sum = 0.0f;
		for(int i=0;i<n;i++){
			sum += fabs(matrix[i*m + col]);
		}
		result[col] = sum;
	}
}

/*
 * Kernel function to reduce a vector to a single value by adding its components
 */
__global__ void vectorReduce_kernel(float* vec, float* result, int n){
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = (i<n) ? vec[i] : 0.0f;
	__syncthreads();
	for (int s= blockDim.x/2;s>0;s>>=1){
		if(tid<s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if(tid==0)
		result[blockIdx.x] = sdata[0];
}

/*
 * Main function
 */
int main(int argc, char ** argv){
	int n=10;
	int m=10;
	bool useRandomSeed = false;
	bool displayTiming = false;
	bool allTime = false;
	double t;

	// Parse command line arguments
	for (int i=1; i<argc; i++){
		if(strcmp(argv[i],"-n") == 0 && i+1 < argc){
			n=atoi(argv[i+1]);
		}else if (strcmp(argv[i],"-m") == 0 && i+1 <argc){
			m=atoi(argv[i+1]);
		}else if (strcmp(argv[i],"-r") == 0 ){
			useRandomSeed = true;
		}else if (strcmp(argv[i],"-t") == 0){
			displayTiming = true;
		}else if (strcmp(argv[i],"-a") == 0){
			allTime = true;
		}
	}

	// Set random seed
	if(useRandomSeed){
		struct timeval myRandom;
		gettimeofday(&myRandom,NULL);
		srand48((int)(myRandom.tv_usec));
	}
	else{
		srand48(123456);
	}

	// Initialize the matrix
	float* matrix = init_matrix(n,m);
	
	/* Only CPU Calculation ==============================*/
	// Calculate row sums
	t = get_time();
	float* rowSums = row_abs_sum(matrix,n,m);
	double time_row = (get_time() - t) * 1000000;

	// Calculate column sums
	t = get_time();
	float* colSums = col_abs_sum(matrix,n,m);
	double time_col = (get_time() - t) * 1000000;

	// Reduce row sums to a single value
	t = get_time();
	float rowSum = vectorReduce(rowSums,n);
	double time_reduce_row = (get_time() - t) * 1000000;

	// Reduce column sums to a single value
	t = get_time();
	float colSum = vectorReduce(colSums,m);
	double time_reduce_col = (get_time() - t) * 1000000;

	// Print results and timing (Serial Version)
	printf("Matrix size: %dx%d\n",n,m);
	printf("Use random seed: %s\n", useRandomSeed ? "true" : "false");
	printf("Print Timing: %s\n", displayTiming ? "true" : "false");
	printf("\n");

	printf("The results for CPU calculation:\n");
	printf("Row sum: %f\n", rowSum);
	printf("Column sum: %f\n", colSum);

	if(displayTiming && !allTime){
		printf("Serial Version (CPU) Time:\n");
		printf("Row sum duration: %lf microseconds\n", time_row);	
		printf("Column sum duration: %lf microseconds\n", time_col);
		printf("Row reduce duration: %lf microseconds\n", time_reduce_row);
		printf("Column reduce duration: %lf microseconds\n", time_reduce_col);
		printf("\n");
	}


	/* CPU and GPU Calculation ====================================== */
	// Allocate memory on CPU and GPU
	float *rowSums_cpu = (float*)malloc(n*sizeof(float));
	float *colSums_cpu = (float*)malloc(m*sizeof(float));
	float rowSum_cpu, colSum_cpu;
	float *result_cpu = (float*)malloc(BLOCK_SIZE * sizeof(float));

	float *matrix_gpu, *rowSums_gpu, *colSums_gpu, *result_gpu;
	cudaMalloc((void**)&matrix_gpu, n*m*sizeof(float));
	cudaMalloc((void**)&rowSums_gpu, n*sizeof(float));
	cudaMalloc((void**)&colSums_gpu, m*sizeof(float));
	cudaMalloc((void **)&result_gpu, BLOCK_SIZE*sizeof(float));

	// Copy matrix to GPU
	cudaMemcpy(matrix_gpu, matrix, n*m*sizeof(float), cudaMemcpyHostToDevice);

	// Calculate row-wise absolute sums on GPU
	t=get_time();
	int num_blocks_row = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	row_abs_sum_kernel<<<num_blocks_row, BLOCK_SIZE>>>(matrix_gpu, rowSums_gpu, n, m);
	cudaMemcpy(rowSums_cpu, rowSums_gpu, n * sizeof(float), cudaMemcpyDeviceToHost);
	double time_row_gpu= (get_time() - t) * 1000000;

	// Calculate column-wise absolute sums on GPU
	t=get_time();
	int num_blocks_col = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
	col_abs_sum_kernel<<<num_blocks_col, BLOCK_SIZE>>>(matrix_gpu, colSums_gpu, n, m);
	cudaMemcpy(colSums_cpu, colSums_gpu, m * sizeof(float), cudaMemcpyDeviceToHost);
	double time_col_gpu= (get_time() - t) * 1000000;

	// Reduce row sums on GPU
	t=get_time();
	int num_blocks_reduce_row = (n + BLOCK_SIZE -1) / BLOCK_SIZE;
	vectorReduce_kernel<<<num_blocks_reduce_row, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(rowSums_gpu, result_gpu, n);
	cudaMemcpy(result_cpu, result_gpu, num_blocks_reduce_row * sizeof(float), cudaMemcpyDeviceToHost);
	rowSum_cpu = 0.0f;
	for (int i=0;i<num_blocks_reduce_row; i++){
		rowSum_cpu += result_cpu[i]; 
	}
	double time_reduce_row_gpu= (get_time() - t) * 1000000;

	// Reduce column sums on GPU
	t=get_time();
	int num_blocks_reduce_col = (m + BLOCK_SIZE -1) / BLOCK_SIZE;
	vectorReduce_kernel<<<num_blocks_reduce_col, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(colSums_gpu, result_gpu, m);
	cudaMemcpy(result_cpu, result_gpu, num_blocks_reduce_col * sizeof(float), cudaMemcpyDeviceToHost);
	colSum_cpu = 0.0f;
	for (int i=0;i<num_blocks_reduce_col; i++){
		colSum_cpu += result_cpu[i];
	}
	double time_reduce_col_gpu= (get_time() - t) * 1000000;

	// Verify results
	printf("The results for CPU and GPU Calculation:\n");
	printf("Row sum: %f\n",rowSum_cpu);
	printf("Column sum: %f\n",colSum_cpu);

	// Print time and compare
	if(displayTiming && allTime){
		printf("\nTime Compare:\n");
		printf("Row sum duration:\nSerial Version: %lf microseconds ; Parallel Version: %lf microseconds\n", time_row, time_row_gpu);	
		printf("Column sum duration:\nSerial Version: %lf microseconds ; Parallel Version: %lf microseconds\n", time_col, time_col_gpu);
		printf("Row reduce duration:\nSerial Version: %lf microseconds ; Parallel Version: %lf microseconds\n", time_reduce_row, time_reduce_row_gpu);
		printf("Column reduce duration:\nSerial Version: %lf microseconds ; Parallel Version: %lf microseconds\n", time_reduce_col, time_reduce_col_gpu);
		printf("\n");
	}

	// Free device memory
	cudaFree(matrix_gpu);
	cudaFree(rowSums_gpu);
	cudaFree(colSums_gpu);
	cudaFree(result_gpu);

	free(matrix);
	free(rowSums);
	free(colSums);
	free(rowSums_cpu);
	free(colSums_cpu);
	free(result_cpu);

	return 0;
}

/*
 * Function to allocate and initialize the matrix.
 */
float* init_matrix(int n, int m){
	float* matrix = (float*)malloc(n * m * sizeof(float));
	for(int i=0;i<n*m;i++){
		matrix[i] = ((float)(drand48())*10.0)-5.0;
	}
	return matrix;
}

/*
 * Function to get the current time.
 */
double get_time(void){
	struct timeval tv;
	double t;

	gettimeofday(&tv, NULL);
	t= tv.tv_sec + (double)tv.tv_usec * 1e-6;

	return t;
}

/*
 * Function to adds together the absolute value of each element of each row.
 * (into a vector of size n)
 */
float* row_abs_sum(float* matrix, int n,int m){
	float* rowSums=(float*)malloc(n*sizeof(float));
	memset(rowSums,0, n*sizeof(float));
	for (int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			rowSums[i] += fabs(matrix[i*m+j]);
		}
	}
	return rowSums;
}

/*
 * Function to adds together the absolute value of each element of each column.
 * (into a vector of size m)
 */
float* col_abs_sum(float* matrix, int n, int m){
	float* colSums=(float*)malloc(m*sizeof(float));
	memset(colSums,0, m*sizeof(float));
	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			colSums[j] += fabs(matrix[i*m+j]);
		}
	}
	return colSums;
}

/*
 * Function to reduce a vector to a single value by adding its components
 */
float vectorReduce(float* vec, int size){
	float sum = 0.0f;
	for (int i=0; i<size; i++){
		sum+=vec[i];
	}
	return sum;
}

