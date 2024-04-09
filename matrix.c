/**
 * @file matrix.c
 * @brief Only CPU calculation to reduce a matrix to a single value
 * 		Code for Task1 of MAP55616 Assignment1.
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

float* init_matrix(int n, int m);
float* row_abs_sum(float* matrix, int n, int m);
float* col_abs_sum(float* matrix, int n, int m);
float vectorReduce(float* vec, int size);
double get_time(void);

int main(int argc, char ** argv){
	int n=10;
	int m=10;
	bool useRandomSeed = false;
	bool displayTiming = false;
	double t;

	// Parse command line arguments
	for (int i=1; i<argc; i++){
		if(strcmp(argv[i],"-n") == 0 && i+1 < argc){
			n=atoi(argv[i+1]);
		}else if (strcmp(argv[i],"-m") == 0 && i+1 <argc){
			m=atoi(argv[i+1]);
		}else if (strcmp(argv[i],"-r") ==0 ){
			useRandomSeed = true;
		}else if (strcmp(argv[i],"-t") ==0){
			displayTiming = true;
		}
	}

	// Set Random seed
	if(useRandomSeed){
		struct timeval myRandom;
		gettimeofday(&myRandom,NULL);
		srand48((int)(myRandom.tv_usec));
	}
	else{
		srand48(123456);
	}

	// Allocate and initialize the matrix
	float* matrix = init_matrix(n,m);

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

	// Print results and timing
	printf("Matrix size: %dx%d\n",n,m);
	printf("Use random seed: %s\n", useRandomSeed ? "true" : "false");
	printf("Print Timing: %s\n", displayTiming ? "true" : "false");
	printf("\n");

	printf("Row sum: %f\n", rowSum);
	printf("Column sum: %f\n", colSum);

	if(displayTiming){
		printf("\n");
		printf("Row sum duration: %lf microseconds\n", time_row);	
		printf("Column sum duration: %lf microseconds\n", time_col);
		printf("Row reduce duration: %lf microseconds\n", time_reduce_row);
		printf("Column reduce duration: %lf microseconds\n", time_reduce_col);
	}

/*	// Check results
	printf("Row sums: ");
        for (int i=0; i<n; i++) {
		printf("%f ", rowSums[i]);
	}
	printf("\n");

	printf("Column sums: ");
	for (int i=0; i<m; i++){
		printf("%f ", colSums[i]);
	}
	printf("\n");
*/

	// Free memory
	free(matrix);
	free(rowSums);
	free(colSums);

	return 0;
}


/*
 * Function to allocate and initialize the matrix.
 */
float* init_matrix(int n, int m){
	float* matrix = (float*)malloc(n*m * sizeof(float));
	for(int i=0;i<n*m;i++){
		matrix[i] = ((float)(drand48())*10.0)-5.0;
	}
	return matrix;
}

/*
 * Function to adds together the absolute value of each element of each row.
 * (into a vector of size n)
 */
float* row_abs_sum(float* matrix, int n,int m){
	float* rowSums = (float*) malloc(n * sizeof(float));
	memset(rowSums, 0, n * sizeof(float));
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
	float* colSums = (float*) malloc(m * sizeof(float));
	memset(colSums, 0, m * sizeof(float));
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
	for (int i = 0; i < size; i++){
		sum+=vec[i];
	}
	return sum;
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
