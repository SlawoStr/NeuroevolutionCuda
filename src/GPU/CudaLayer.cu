#include "CudaLayer.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "src/Utility/CudaError.h"

template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}


template<typename Func>
__global__ void feedForwardCudaOTP(float* input, float* C, float* output, size_t inNeuronPerModel, size_t connectionNumber, size_t outLayerSize, Func activationFunction)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < outLayerSize)
	{
		int modelID = tid / connectionNumber;
		int neuronOffset = modelID * inNeuronPerModel;
		int weightOffset = tid * (inNeuronPerModel + 1);	
		float sum{};
		int i{};
		for (i = 0; i < inNeuronPerModel; ++i)
		{
			sum += input[neuronOffset + i] * C[weightOffset + i];
		}
		// Add bias weigh
		sum += C[weightOffset + i];
		// Use Activation function
		output[tid] = activationFunction(sum);
	}
}

/*
template<typename Func>
__global__ void feedForwardCudaOTP2(float* input, float* C, float* output, size_t inNeuronPerModel, size_t connectionNumber, size_t outLayerSize, Func activationFunction,int allConnNumber)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < outLayerSize)
	{
		//int modelID = tid / connectionNumber;
		int neuronOffset = 0;
		float sum{};
		int i{};

		for (i = blockIdx.x * blockDim.x + threadIdx.x; i < allConnNumber; i += blockDim.x * gridDim.x)
		{
			sum += input[neuronOffset++] * C[i];
		}
		// Add bias weigh
		sum += C[i];
		// Use Activation function
		output[tid] = activationFunction(sum);
	}
}
*/

template<unsigned int blockSize,typename Func>
__global__ void feedForwardGrid(float* input, float* C, float* output, size_t inNeuronPerModel, size_t connectionNumber, size_t outLayerSize, Func activationFunction)
{
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	int modelID = blockIdx.x / connectionNumber;
	int neuronOffset = modelID * inNeuronPerModel;
	int weightOffset = blockIdx.x * (inNeuronPerModel + 1);

	sdata[tid] = input[neuronOffset + tid] * C[weightOffset + tid];
	for (int i = tid + blockDim.x; i < inNeuronPerModel; i += blockDim.x)
	{
		sdata[tid] += input[neuronOffset + i] * C[weightOffset + i];
	}
	__syncthreads();
	// Reduction
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

	if (tid < 32) warpReduce<blockSize>(sdata, tid);
	// Copy result from shared memory
	if (tid == 0)
	{
		output[blockIdx.x] = activationFunction(sdata[0] + C[weightOffset + inNeuronPerModel]);
	}
}

__global__ void mapModelWeightTo(float* src, float* des, int connectionPerModel, int modelWeight, int taskSize)
{
	int srcOffset = blockIdx.x * connectionPerModel;
	int desOffset = blockIdx.x * modelWeight;
	for (int i = threadIdx.x; i < connectionPerModel; i += blockDim.x)
	{
		des[desOffset + i] = src[srcOffset + i];
	}
}

__global__ void mapModelWeightFrom(float* src, const float* des, int connectionPerModel, int modelWeight, int taskSize)
{
	int srcOffset = blockIdx.x * connectionPerModel;
	int desOffset = blockIdx.x * modelWeight;
	for (int i = threadIdx.x; i < connectionPerModel; i += blockDim.x)
	{
		src[srcOffset + i] = des[desOffset + i];
	}
}

template<typename Func>
void runFeedForwardKernel(float* input, float* C, float* output, size_t inNeuronPerModel, size_t connectionNumber, size_t outLayerSize, Func activationFunction, unsigned threadNumber, size_t modelNumber)
{
	if (inNeuronPerModel < 32)
	{
		
		//feedForwardCudaOTP << <(static_cast<unsigned>(outLayerSize + threadNumber - 1)) / threadNumber, threadNumber >> >
		//	(input, C, output, inNeuronPerModel, connectionNumber, outLayerSize, activationFunction);

		//feedForwardCudaOTP2 << <(static_cast<unsigned>(outLayerSize + threadNumber - 1)) / threadNumber, threadNumber >> >
		//	(input, C, output, inNeuronPerModel, connectionNumber, outLayerSize, activationFunction, modelNumber * connectionNumber);
	}
	else
	{
		unsigned sharedMemory = 32;
		while (sharedMemory * 2 < inNeuronPerModel && sharedMemory * 2 <= 1024 && sharedMemory * 2 <= threadNumber)
		{
			sharedMemory *= 2;
		}

		switch (sharedMemory)
		{
		case 1024:
			feedForwardGrid<1024> << <static_cast<unsigned>(outLayerSize), sharedMemory, sharedMemory * sizeof(float) >> >
				(input, C, output, inNeuronPerModel, connectionNumber, outLayerSize, activationFunction);
			break;
		case 512:
			feedForwardGrid<512> << <static_cast<unsigned>(outLayerSize), sharedMemory, sharedMemory * sizeof(float) >> >
				(input, C, output, inNeuronPerModel, connectionNumber, outLayerSize, activationFunction);
			break;
		case 256:
			feedForwardGrid<256> << <static_cast<unsigned>(outLayerSize), sharedMemory, sharedMemory * sizeof(float) >> >
				(input, C, output, inNeuronPerModel, connectionNumber, outLayerSize, activationFunction);
			break;
		case 128:
			feedForwardGrid<128> << <static_cast<unsigned>(outLayerSize), sharedMemory, sharedMemory * sizeof(float) >> >
				(input, C, output, inNeuronPerModel, connectionNumber, outLayerSize, activationFunction);
			break;
		case 64:
			feedForwardGrid<64> << <static_cast<unsigned>(outLayerSize), sharedMemory, sharedMemory * sizeof(float) >> >
				(input, C, output, inNeuronPerModel, connectionNumber, outLayerSize, activationFunction);
			break;
		case 32:
			feedForwardGrid<32> << <static_cast<unsigned>(outLayerSize), sharedMemory, sharedMemory * sizeof(float) >> >
				(input, C, output, inNeuronPerModel, connectionNumber, outLayerSize, activationFunction);
			break;
		}
	}
}


CudaLayer::CudaLayer(size_t neuronNumber, size_t connectionNumber, size_t modelNumber, ActivationFunction actFunc)
	:m_neuronPerModel{ neuronNumber }, m_connectionPerNeuron{ connectionNumber }, m_modelNumber{ modelNumber }, m_actFunc{ actFunc }
{
	initLayer();
}

CudaLayer::CudaLayer(size_t neuronNumber, size_t connectionNumber, size_t modelNumber)
	:m_neuronPerModel{ neuronNumber }, m_connectionPerNeuron{ connectionNumber }, m_modelNumber{ modelNumber }, m_actFunc{ ActivationFunction::NONE }
{
	initLayer();
}

void CudaLayer::setNeurons(const float* input, bool isHostAllocated)
{
	if (isHostAllocated)
	{
		cudaMemcpy(thrust::raw_pointer_cast(md_neurons.data()), input, sizeof(float) * md_neurons.size(), cudaMemcpyHostToDevice);
	}
	else
	{
		cudaMemcpy(thrust::raw_pointer_cast(md_neurons.data()), input, sizeof(float) * md_neurons.size(), cudaMemcpyDeviceToDevice);
	}
}

void CudaLayer::getNeurons(float* output, bool isHostAllocated)
{
	if (isHostAllocated)
	{
		cudaMemcpy(output, thrust::raw_pointer_cast(md_neurons.data()), sizeof(float) * md_neurons.size(), cudaMemcpyDeviceToHost);
	}
	else
	{
		cudaMemcpy(output, thrust::raw_pointer_cast(md_neurons.data()), sizeof(float) * md_neurons.size(), cudaMemcpyDeviceToDevice);
	}
}

void CudaLayer::setWeight(const float* input, bool isHostAllocated)
{
	if (isHostAllocated)
	{
		cudaMemcpy(thrust::raw_pointer_cast(md_weights.data()), input, sizeof(float) * md_weights.size(), cudaMemcpyHostToDevice);
	}
	else
	{
		cudaMemcpy(thrust::raw_pointer_cast(md_weights.data()), input, sizeof(float) * md_weights.size(), cudaMemcpyDeviceToDevice);
	}
}

void CudaLayer::getWeight(float* output, bool isHostAllocated)
{
	if (isHostAllocated)
	{
		cudaMemcpy(output, thrust::raw_pointer_cast(md_weights.data()), sizeof(float) * md_weights.size(), cudaMemcpyDeviceToHost);
	}
	else
	{
		cudaMemcpy(output, thrust::raw_pointer_cast(md_weights.data()), sizeof(float) * md_weights.size(), cudaMemcpyDeviceToDevice);
	}
}

void CudaLayer::setConnections(size_t connectionNumber)
{
	m_connectionPerNeuron = connectionNumber;
	md_weights.resize((m_neuronPerModel + 1) * m_connectionPerNeuron * m_modelNumber);
}

void CudaLayer::feedForward(CudaLayer* next, unsigned threadNumber)
{
	size_t outLayerSize = next->getLayerSize();
	switch (next->m_actFunc)
	{
		case ActivationFunction::RELU:
		{
			auto actFunc = [=] __device__(float x) {
				return x <= 0 ? 0 : x;
			};
			runFeedForwardKernel(thrust::raw_pointer_cast(md_neurons.data()), thrust::raw_pointer_cast(md_weights.data()), thrust::raw_pointer_cast(next->md_neurons.data()), m_neuronPerModel, m_connectionPerNeuron, outLayerSize, actFunc, threadNumber,m_modelNumber);
			break;
		}
		case ActivationFunction::TANH:
		{
			auto actFunc = [=] __device__(float x) {
				return tanh(x);
			};
			runFeedForwardKernel(thrust::raw_pointer_cast(md_neurons.data()), thrust::raw_pointer_cast(md_weights.data()), thrust::raw_pointer_cast(next->md_neurons.data()), m_neuronPerModel, m_connectionPerNeuron, outLayerSize, actFunc, threadNumber, m_modelNumber);
			break;
		}
		case ActivationFunction::SIGMOID:
		{
			auto actFunc = [=] __device__(float x) {
				return 1.0f / (1.0f + exp(-x));
			};
			runFeedForwardKernel(thrust::raw_pointer_cast(md_neurons.data()), thrust::raw_pointer_cast(md_weights.data()), thrust::raw_pointer_cast(next->md_neurons.data()), m_neuronPerModel, m_connectionPerNeuron, outLayerSize, actFunc, threadNumber, m_modelNumber);
			break;
		}
	}
}

size_t CudaLayer::getLayerSize() const
{
	return md_neurons.size();
}

void CudaLayer::getModelWeight(float* output, size_t modelOffset, unsigned threadNumber)
{
	int connectionPerModel = (m_neuronPerModel + 1) * m_connectionPerNeuron;
	mapModelWeightTo << <m_modelNumber, threadNumber >> > (thrust::raw_pointer_cast(md_weights.data()), output, connectionPerModel, modelOffset, m_modelNumber);
}

void CudaLayer::setModelWeight(const float* input, size_t modelOffset,unsigned threadNumber)
{
	int connectionPerModel = (m_neuronPerModel + 1) * m_connectionPerNeuron;
	mapModelWeightFrom << <m_modelNumber, threadNumber >> > (thrust::raw_pointer_cast(md_weights.data()), input, connectionPerModel, modelOffset, m_modelNumber);
}

void CudaLayer::initLayer()
{
	// Set up neurons
	size_t networkSize = m_neuronPerModel * m_modelNumber;
	md_neurons.resize(networkSize);
	// Set up weights
	if (m_connectionPerNeuron > 0)
	{
		md_weights.resize((m_neuronPerModel + 1) * m_connectionPerNeuron * m_modelNumber);
	}
}