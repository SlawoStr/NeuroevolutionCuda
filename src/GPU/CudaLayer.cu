#include "CudaLayer.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "src/Utility/CudaError.h"
#include "src/Utility/Timer.h"

template<typename Func>
__global__ void feedForwardOptimized(float* input, float* C, float* output, int inNeuronPerModel, int connectionNumber, int weightPerKernel, int weightNumber, int stride, Func activationFunction)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// Index of first neuron in model
	int neuronIndex = tid / connectionNumber * inNeuronPerModel;
	// Weight loop settings
	int outNeuronNumber = blockIdx.x != (gridDim.x - 1) ? blockDim.x : stride;
	int startIndex = blockIdx.x * weightPerKernel + threadIdx.x;
	int endIndex = blockIdx.x == (gridDim.x - 1) ? weightNumber : (blockIdx.x + 1) * weightPerKernel;
	// Dont process last batch of connectons (bias connection)
	endIndex -= outNeuronNumber;
	// Sum connections
	float sum{};
	int i{};
	for (i = startIndex; i < endIndex; i += outNeuronNumber)
	{
		sum += input[neuronIndex++] * C[i];
	}
	// Add bias weigh
	sum += C[i];
	// Use Activation function
	output[tid] = activationFunction(sum);
}

__global__ void mapModelWeightToSharedBatch(float* src, float* des, int inNeuronPerModel, int connectionNumber, int weightPerKernel, int weightNumber, int stride, int weightPerModel, int batchSize)
{
	extern __shared__ float sdata[];

	// Index of weight that belong to this block
	int startIndex = blockIdx.x * weightPerKernel + threadIdx.x;
	int endIndex = blockIdx.x == (gridDim.x - 1) ? weightNumber : (blockIdx.x + 1) * weightPerKernel;
	// Number of data batches to process
	int elementPerBatch = blockIdx.x != (gridDim.x - 1) ? batchSize * blockDim.x : batchSize * stride;
	// Number of neurons that will be processed by this block
	int outNeuronNumber = blockIdx.x != (gridDim.x - 1) ? blockDim.x : stride;
	// Number of neurons that were already added
	int weightOffset = threadIdx.x;
	int loopCounter{ 1 };
	// For each batch
	for (int i = startIndex, lastBatchElement = startIndex + elementPerBatch; i < endIndex; i += elementPerBatch, lastBatchElement += elementPerBatch)
	{
		if (lastBatchElement > endIndex)
		{
			lastBatchElement = endIndex;
		}
		// Copy batch to shared memory
		for (int j = i, sIndex = threadIdx.x; j < lastBatchElement; j += blockDim.x, sIndex += blockDim.x)
		{
			sdata[sIndex] = src[j];
		}
		__syncthreads();
		int modelID = (blockIdx.x * blockDim.x) / connectionNumber;
		int neuronID = (blockIdx.x * blockDim.x) % connectionNumber;
		int destOffset = modelID * weightPerModel + neuronID * inNeuronPerModel;
		int batchNeuron = loopCounter * batchSize;
		int modelNeuronCounter{};
		if (batchNeuron > inNeuronPerModel)
		{
			batchNeuron = inNeuronPerModel - (loopCounter - 1) * batchSize;
		}
		else
		{
			batchNeuron = batchSize;
		}
		for (int j = 0; j < outNeuronNumber; ++j)
		{
			int tempModelID = (blockIdx.x * blockDim.x + j) / connectionNumber;
			if (modelID != tempModelID)
			{
				modelID = tempModelID;
				destOffset = modelID * weightPerModel;
				modelNeuronCounter = 0;
			}
			int tempOffset = weightOffset;
			for (int k = threadIdx.x; k < batchNeuron; k += blockDim.x)
			{
				des[destOffset + modelNeuronCounter * inNeuronPerModel + tempOffset] = sdata[j + k * outNeuronNumber];
				tempOffset += blockDim.x;
			}
			modelNeuronCounter++;
		}
		__syncthreads();
		weightOffset += batchSize;
		loopCounter++;
	}
}

__global__ void mapModelWeightFromSharedBatch(float* src, const float* des, int inNeuronPerModel, int connectionNumber, int weightPerKernel, int weightNumber, int stride, int weightPerModel, int batchSize)
{
	extern __shared__ float sdata[];
	// Index of weight that belong to this block
	int startIndex = blockIdx.x * weightPerKernel + threadIdx.x;
	int endIndex = blockIdx.x == (gridDim.x - 1) ? weightNumber : (blockIdx.x + 1) * weightPerKernel;
	// Number of data batches to process
	int elementPerBatch = blockIdx.x != (gridDim.x - 1) ? batchSize * blockDim.x : batchSize * stride;
	// Number of neurons that will be processed by this block
	int outNeuronNumber = blockIdx.x != (gridDim.x - 1) ? blockDim.x : stride;
	// Number of neurons that were already added
	int weightOffset = threadIdx.x;
	int loopCounter{ 1 };
	// For each batch
	for (int i = startIndex, lastBatchElement = startIndex + elementPerBatch; i < endIndex; i += elementPerBatch, lastBatchElement += elementPerBatch)
	{
		if (lastBatchElement > endIndex)
		{
			lastBatchElement = endIndex;
		}
		int modelID = (blockIdx.x * blockDim.x) / connectionNumber;
		int neuronID = (blockIdx.x * blockDim.x) % connectionNumber;
		int destOffset = modelID * weightPerModel + neuronID * inNeuronPerModel;
		int batchNeuron = loopCounter * batchSize;
		int modelNeuronCounter{};
		if (batchNeuron > inNeuronPerModel)
		{
			batchNeuron = inNeuronPerModel - (loopCounter - 1) * batchSize;
		}
		else
		{
			batchNeuron = batchSize;
		}
		__syncthreads();
		for (int j = 0; j < outNeuronNumber; ++j)
		{
			int tempModelID = (blockIdx.x * blockDim.x + j) / connectionNumber;
			if (modelID != tempModelID)
			{
				modelID = tempModelID;
				destOffset = modelID * weightPerModel;
				modelNeuronCounter = 0;
			}
			int tempOffset = weightOffset;
			for (int k = threadIdx.x; k < batchNeuron; k += blockDim.x)
			{
				sdata[j + k * outNeuronNumber] = des[destOffset + modelNeuronCounter * inNeuronPerModel + tempOffset];
				tempOffset += blockDim.x;
			}
			modelNeuronCounter++;
		}
		__syncthreads();
		// Copy batch to shared memory
		for (int j = i, sIndex = threadIdx.x; j < lastBatchElement; j += blockDim.x, sIndex += blockDim.x)
		{
			src[j] = sdata[sIndex];
		}
		weightOffset += batchSize;
		loopCounter++;
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

void CudaLayer::setConnections(size_t connectionNumber)
{
	// Create weights
	m_connectionPerNeuron = connectionNumber;
	md_weights.resize((m_neuronPerModel + 1) * m_connectionPerNeuron * m_modelNumber);
	// Get number of blocks for kernels
	m_blockNumber = static_cast<int>((m_modelNumber * m_connectionPerNeuron + m_threadNumber - 1)) / m_threadNumber;
	// Get number of neurons to process for last kernel
	m_stride = m_modelNumber * m_connectionPerNeuron % m_threadNumber;
	if (m_stride == 0)
	{
		m_stride = m_threadNumber;
	}
}

void CudaLayer::feedForward(CudaLayer* next)
{
	switch (next->m_actFunc)
	{
		case ActivationFunction::RELU:
		{
			auto actFunc = [=] __device__(float x) {
				return x <= 0 ? 0 : x;
			};
			feedForwardOptimized << <m_blockNumber, m_threadNumber >> > (thrust::raw_pointer_cast(md_neurons.data()), thrust::raw_pointer_cast(md_weights.data()), thrust::raw_pointer_cast(next->md_neurons.data()),
				m_neuronPerModel, m_connectionPerNeuron, m_weightPerBlock, md_weights.size(),m_stride, actFunc);
			break;
		}
		case ActivationFunction::TANH:
		{
			auto actFunc = [=] __device__(float x) {
				return tanh(x);
			};
			feedForwardOptimized << <m_blockNumber, m_threadNumber >> > (thrust::raw_pointer_cast(md_neurons.data()), thrust::raw_pointer_cast(md_weights.data()), thrust::raw_pointer_cast(next->md_neurons.data()),
				m_neuronPerModel, m_connectionPerNeuron, m_weightPerBlock, md_weights.size(),m_stride, actFunc);
			break;
		}
		case ActivationFunction::SIGMOID:
		{
			auto actFunc = [=] __device__(float x) {
				return 1.0f / (1.0f + exp(-x));
			};
			feedForwardOptimized << <m_blockNumber, m_threadNumber >> > (thrust::raw_pointer_cast(md_neurons.data()), thrust::raw_pointer_cast(md_weights.data()), thrust::raw_pointer_cast(next->md_neurons.data()),
				m_neuronPerModel, m_connectionPerNeuron, m_weightPerBlock, md_weights.size(),m_stride, actFunc);
			break;
		}
	}
}

size_t CudaLayer::getLayerSize() const
{
	return md_neurons.size();
}

void CudaLayer::getModelWeight(float* output, size_t modelOffset)
{
	mapModelWeightToSharedBatch << < m_blockNumber, m_threadNumber, sizeof(float)* m_threadNumber* m_batchSize >> > (thrust::raw_pointer_cast(md_weights.data()), output,
		m_neuronPerModel + 1, m_connectionPerNeuron, m_weightPerBlock, md_weights.size(), m_stride, modelOffset, m_batchSize);
}

void CudaLayer::setModelWeight(const float* input, size_t modelOffset)
{
	mapModelWeightFromSharedBatch << < m_blockNumber, m_threadNumber, sizeof(float)* m_threadNumber* m_batchSize >> > (thrust::raw_pointer_cast(md_weights.data()), input,
		m_neuronPerModel + 1, m_connectionPerNeuron, m_weightPerBlock, md_weights.size(), m_stride, modelOffset, m_batchSize);
}

void CudaLayer::initLayer()
{
	// Set up neurons
	md_neurons.resize(m_neuronPerModel * m_modelNumber);
	// Set up weights
	setConnections(m_connectionPerNeuron);
	m_weightPerBlock = static_cast<int>(m_threadNumber * (m_neuronPerModel + 1));
}