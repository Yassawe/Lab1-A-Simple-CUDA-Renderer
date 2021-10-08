#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <cmath>

#include <cuda_runtime.h>

// helpful macros
#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}


#define CheckCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      assert(0);                                                        \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define BLOCK_SIZE	32


void computeRefMatrixMul(float *C, const float *A, const float *B, unsigned int height_A, unsigned int width_A, unsigned int width_B) {
	for(unsigned int i=0; i<height_A; i++) {
		for(unsigned int j=0; j<width_B; j++) {
			double sum = 0;
      for(unsigned int k=0; k<width_A; k++) {
				double a = A[(i*width_A)+k];
				double b = B[(k*width_B)+j];
				sum += a*b;
      }
      C[(i*width_B)+j] = (float)sum;
    }
	}
}

__global__ 
void matrixMul_naive(float* C, float* A, float* B, int wA, int wB) {
	// TODO: fill me
}

__global__ 
void matrixMul_shmem( float* C, float* A, float* B, int wA, int wB)
{
	// TODO: fill me
}

void randomInitialization(float *data, int size) {
	srand(time(NULL));
	for(int i=0; i<size; i++) {
		data[i] = rand()/(float)RAND_MAX;
  }
}

bool compareArray(const float *reference, const float *data, const unsigned int len, const float epsilon) {
  assert(epsilon >= 0);
  float error = 0;
  float ref = 0;
  
	for(unsigned int i=0; i<len; i++) {
    float diff = reference[i] - data[i];
    error += diff * diff;
    ref += reference[i] * reference[i];
  }
  float normRef = sqrtf(ref);

  if (fabs(ref) < 1e-7) {
#ifdef _DEBUG
		std::cerr << "ERROR, reference l2-norm is 0\n";
#endif
      return false;
  }
  float normError = sqrtf(error);
  error = normError / normRef;
  bool result = error < epsilon;
#ifdef _DEBUG
  if (! result) {
      std::cerr<<"ERROR, l2-norm error "<<error<<" is greater than epsilon "<<epsilon<<"\n";
  }
#endif
  return result;
}

int matrixMul(int block_size, dim3 &dimA, dim3 &dimB)
{
  // Allocate host memory for matrices A
  unsigned int size_A		= dimA.x*dimA.y;
  unsigned int mem_size_A = sizeof(float)*size_A;
  float *h_A = (float*)malloc(mem_size_A);

  // Allocate host memory for matrices B
  unsigned int size_B		= dimB.x*dimB.y;
  unsigned int mem_size_B = sizeof(float)*size_B;
  float *h_B = (float*)malloc(mem_size_B);

  // Allocate host matrix C
  dim3 dimC(dimB.x, dimA.y, 1);
  unsigned int mem_size_C = dimC.x*dimC.y*sizeof(float);
  unsigned int size_C		= dimC.x*dimC.y;
  float *h_C = (float*)malloc(mem_size_C);

  // Initialize host memory A & B 
  randomInitialization(h_A, size_A);
  randomInitialization(h_B, size_B);
  randomInitialization(h_C, size_C);

  // compute gold solution
  printf("\n[Step-1] Computing reference result using host-side CPU ... ");
  float *reference_C = (float *)malloc(mem_size_C);
  computeRefMatrixMul(reference_C, h_A, h_B, dimA.y, dimA.x, dimB.x);
  printf("DONE!\n");

  // Allocate device memory for A, B, and C
  float *d_A, *d_B, *d_C;
  CheckCudaErrors(cudaMalloc((void **)&d_A, mem_size_A));
	CheckCudaErrors(cudaMalloc((void **)&d_B, mem_size_B));
  CheckCudaErrors(cudaMalloc((void **)&d_C, mem_size_C));

  // copy host-side A and B to device
  CheckCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
	CheckCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
	CheckCudaErrors(cudaMemcpy(d_C, h_C, mem_size_B, cudaMemcpyHostToDevice));

  // Setup execution parameters
  dim3 threads(block_size, block_size);
  dim3 grid((dimB.x/threads.x), (dimA.y/threads.y));
    
	//----------------------------------------------------------
	// Part A. Naive implementation of matrix-multiplication
	//----------------------------------------------------------
  // Allocate CUDA events that is used for measuring kernel execution latency
  cudaEvent_t start, stop;
  CheckCudaErrors(cudaEventCreate(&start));
  CheckCudaErrors(cudaEventCreate(&stop));
    
	// For accurate performance measurements, perform a dummy kernel launch for warm-up
  matrixMul_naive<<< grid, threads >>>(d_C, d_A, d_B, dimA.x, dimB.x);
  cudaDeviceSynchronize();
  
  printf("\n[Step-2] Computing result using naive version of CUDA kernel ... ");

  // Record the start event
  CheckCudaErrors(cudaEventRecord(start, NULL));

  // Execute the kernel
  int nIter = 500;
  for(int j=0; j<nIter; j++) {
		matrixMul_naive<<<grid, threads>>>(d_C, d_A, d_B, dimA.x, dimB.x);
  }

  // Record the stop event
  CheckCudaErrors(cudaEventRecord(stop, NULL));
  // Wait for the stop event to be finalized
  CheckCudaErrors(cudaEventSynchronize(stop));
  printf("DONE!\n");

	// measure average latency incurred for this kernel execution
  float msecTotal = 0.0f;
  CheckCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul		= msecTotal/nIter;
  double flopsPerMatrixMul	= 2.0*(double)dimA.x*(double)dimA.y*(double)dimB.x;
  double gigaFlops			= (flopsPerMatrixMul*1.0e-9f)/(msecPerMatrixMul/1000.0f);
  printf("- Math Size = %.0f OPs\n", flopsPerMatrixMul);
  printf("- Performance = %.2f GFLOP/sec (Time = %.3f msec)\n", gigaFlops, msecPerMatrixMul); 

  // Copy result from device to host
  CheckCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

  printf("- Correctness (reference vs. CUDA): ");
	bool correct = compareArray(reference_C, h_C, size_C, 1.0e-6f); 
  // check result
	if(correct != true) {
		printf("%s\n", "FAIL");
	}
	else {
		printf("%s\n", "PASS");
	}

	//----------------------------------------------------------
	// Part B. Better implementation of matrix-multiplication
	//----------------------------------------------------------
  // Initialize host memory A & B 
  randomInitialization(h_A, size_A);
  randomInitialization(h_B, size_B);
  randomInitialization(h_C, size_C);
  // compute gold solution
  computeRefMatrixMul(reference_C, h_A, h_B, dimA.y, dimA.x, dimB.x);

  // copy host-side A and B to device
  CheckCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
	CheckCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
	CheckCudaErrors(cudaMemcpy(d_C, h_C, mem_size_B, cudaMemcpyHostToDevice));

	// For accurate performance measurements, perform a dummy kernel launch for warm-up
  matrixMul_shmem<<< grid, threads >>>(d_C, d_A, d_B, dimA.x, dimB.x);
  cudaDeviceSynchronize();
  
  printf("\n[Step-3] Computing result using shmem version of CUDA kernel ... ");

  // Record the start event
  CheckCudaErrors(cudaEventRecord(start, NULL));

  // Execute the kernel
  nIter = 500;
  for(int j=0; j<nIter; j++) {
		matrixMul_shmem<<<grid, threads>>>(d_C, d_A, d_B, dimA.x, dimB.x);
  }

  // Record the stop event
  CheckCudaErrors(cudaEventRecord(stop, NULL));
  // Wait for the stop event to be finalized
  CheckCudaErrors(cudaEventSynchronize(stop));
  printf("DONE!\n");

	// measure average latency incurred for this kernel execution
  msecTotal = 0.0f;
  CheckCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  msecPerMatrixMul	= msecTotal/nIter;
  flopsPerMatrixMul	= 2.0*(double)dimA.x*(double)dimA.y*(double)dimB.x;
  gigaFlops			= (flopsPerMatrixMul*1.0e-9f)/(msecPerMatrixMul/1000.0f);
  printf("- Math Size = %.0f OPs\n", flopsPerMatrixMul);
  printf("- Performance = %.2f GFLOP/sec (Time = %.3f msec)\n", gigaFlops, msecPerMatrixMul); 

  // Copy result from device to host
  CheckCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

  printf("- Correctness (reference vs. CUDA): ");
	correct = compareArray(reference_C, h_C, size_C, 1.0e-6f); 
  // check result
	if(correct != true) {
		printf("%s\n", "FAIL");
	}
	else {
		printf("%s\n", "PASS");
	}

  // Clean up memory
  free(h_A);
  free(h_B);
  free(h_C);
  CheckCudaErrors(cudaFree(d_A));
  CheckCudaErrors(cudaFree(d_B));
	CheckCudaErrors(cudaFree(d_C));
	//CheckCudaErrors(cudaFree(d_C));	// This should cause an error ...

  if(correct) {
      return EXIT_SUCCESS;
  }
  else
  {
      return EXIT_FAILURE;
  }
}

// main
int main(int argc, char **argv)
{
	printf("\n---------------------------------------------------\n");
  printf("[Lab 1] Part 2: Matrix-Multiplication Using CUDA\n");
	printf("---------------------------------------------------\n");

	// dimension of matrix A and B
  dim3 dimA(10*BLOCK_SIZE, 10*BLOCK_SIZE, 1);
  dim3 dimB(20*BLOCK_SIZE, 10*BLOCK_SIZE, 1);

	// check if dimension of A & B match properly
  if(dimA.x!=dimB.y) {
	printf("Error: outer matrix dimensions must be equal. (%d != %d)\n", dimA.x, dimB.y);
	exit(EXIT_FAILURE);
  }
	// target matrix configuration
  printf("\n- MatrixA(%d,%d), MatrixB(%d,%d)\n", dimA.x, dimA.y, dimB.x, dimB.y);

	// do matrix multiplication
  exit(matrixMul(BLOCK_SIZE, dimA, dimB));
}
