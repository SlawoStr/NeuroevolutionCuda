#include "CudaRNG.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaError.h"

////////////////////////////////////////////////////////////
__global__ void kernelInitRandomState(unsigned seed, curandState* state)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, tid, 0, &state[tid]);
}

////////////////////////////////////////////////////////////
__device__ int randomInt(curandState& state, int minValue, int maxValue)
{
	maxValue -= 1;
	float randomValue = curand_uniform(&state);
	randomValue *= (maxValue - minValue + 0.999999);
	randomValue += minValue;
	return (int)truncf(randomValue);
}

////////////////////////////////////////////////////////////
__device__ float randomFloat(curandState& state, float minValue, float maxValue)
{
	float randomValue = curand_uniform(&state);
	return randomValue * (maxValue - minValue) + minValue;
}

////////////////////////////////////////////////////////////
__global__ void kernelRandomInt(int minValue, int maxValue, size_t size, int* d_arr, curandState* state)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	// Copy state to local memory for efficiency
	curandState localState = state[tid];
	for (int i = tid; i < size; i += stride)
	{
		d_arr[i] = randomInt(localState, minValue, maxValue);
	}
	// Copy state back to global memory
	state[tid] = localState;
}

////////////////////////////////////////////////////////////
__global__ void kernelRandomFloat(float minValue, float maxValue, size_t size, float* d_arr, curandState* state)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	// Copy state to local memory for efficiency
	curandState localState = state[tid];
	// Generate pseudo-random floats
	for (int i = tid; i < size; i += stride)
	{
		d_arr[i] = randomFloat(localState, minValue, maxValue);
	}
	// Copy state back to global memory
	state[tid] = localState;
}

////////////////////////////////////////////////////////////
curandState* generateRandomStates(unsigned blockNumber, unsigned threadNumber, unsigned seed)
{
	curandState* state{ nullptr };
	checkCudaErrors(cudaMalloc(&state, blockNumber * threadNumber * sizeof(curandState)));
	kernelInitRandomState << <blockNumber, threadNumber >> > (seed, state);
	checkCudaErrors(cudaPeekAtLastError());
	return state;
}

////////////////////////////////////////////////////////////
void randomFloat(unsigned blockNumber, unsigned threadNumber, float min, float max, size_t size, float* d_arr, curandState* state)
{
	kernelRandomFloat << < blockNumber, threadNumber >> > (min, max, size, d_arr, state);
	checkCudaErrors(cudaPeekAtLastError());
}

////////////////////////////////////////////////////////////
float* randomFloat(unsigned blockNumber, unsigned threadNumber, float min, float max, size_t size, curandState* state)
{
	float* d_arr{ nullptr };
	checkCudaErrors(cudaMalloc(&d_arr, sizeof(float) * size));
	kernelRandomFloat << < blockNumber, threadNumber >> > (min, max, size, d_arr, state);
	checkCudaErrors(cudaPeekAtLastError());
	return d_arr;
}

////////////////////////////////////////////////////////////
void randomInt(unsigned blockNumber, unsigned threadNumber, int min, int max, size_t size, int* d_arr, curandState* state)
{
	kernelRandomInt << < blockNumber, threadNumber >> > (min, max, size, d_arr, state);
	checkCudaErrors(cudaPeekAtLastError());
}

////////////////////////////////////////////////////////////
int* randomInt(unsigned blockNumber, unsigned threadNumber, int min, int max, size_t size, curandState* state)
{
	int* d_arr{ nullptr };
	checkCudaErrors(cudaMalloc(&d_arr, sizeof(int) * size));
	kernelRandomInt << < blockNumber, threadNumber >> > (min, max, size, d_arr, state);
	checkCudaErrors(cudaPeekAtLastError());
	return d_arr;
}
