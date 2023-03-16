#include "CudaUniformCrossover.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void crossoverUniformKernel(float* weight, float crossoverProbability, int weightPerModel, int taskNumber, curandState* state)
{
	__shared__ float crossProba;

	for (int i = blockIdx.x; i < taskNumber; i += gridDim.x)
	{
		if (threadIdx.x == 0)
		{
			crossProba = randomFloat(state[threadIdx.x + blockIdx.x * blockDim.x], 0.0f, 1.0f);
		}
		__syncthreads();
		if (crossProba < crossoverProbability)
		{
			int lhsOffset = i * 2 * weightPerModel;
			int rhsOffset = (i * 2 + 1) * weightPerModel;
			for (int j = threadIdx.x; j < weightPerModel; j += blockDim.x)
			{
				if (randomFloat(state[threadIdx.x + blockIdx.x * blockDim.x], 0.0f, 1.0f) < 0.5f)
				{
					float value = weight[lhsOffset + j];
					weight[lhsOffset + j] = weight[rhsOffset + j];
					weight[rhsOffset + j] = value;
				}
			}
		}
		__syncthreads();
	}
}

void CudaUniformCrossover::runCrossover(thrust::device_vector<float>& newWeights, int weightPerModel, int parentPairNumber, curandState* state, unsigned blockNumber, unsigned threadNumber)
{
	int optBlockNumber = std::min(static_cast<unsigned>(parentPairNumber), blockNumber);
	crossoverUniformKernel << < optBlockNumber, threadNumber >> > (thrust::raw_pointer_cast(newWeights.data()), m_crossProbability, weightPerModel, parentPairNumber, state);
}