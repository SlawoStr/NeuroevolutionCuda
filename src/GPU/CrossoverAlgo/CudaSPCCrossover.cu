#include "CudaSPCCrossover.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void crossoverSPCKernel(float* weight, float crossoverProbability, int weightPerModel, int taskNumber, curandState* state)
{
	__shared__ int crossPoint;
	__shared__ float crossProba;

	for (int i = blockIdx.x; i < taskNumber; i += gridDim.x)
	{
		if (threadIdx.x == 0)
		{
			crossProba = randomFloat(state[blockIdx.x], 0.0f, 1.0f);
			crossPoint = randomFloat(state[blockIdx.x], 1, weightPerModel - 1);
		}
		__syncthreads();

		if (crossProba < crossoverProbability)
		{
			int lhsOffset = i * 2 * weightPerModel;
			int rhsOffset = (i * 2 + 1) * weightPerModel;
			for (int j = threadIdx.x; j < crossPoint; j += blockDim.x)
			{
				float value = weight[lhsOffset + j];
				weight[lhsOffset + j] = weight[rhsOffset + j];
				weight[rhsOffset + j] = value;
			}
		}
		__syncthreads();
	}
}

void CudaSPCCrossover::runCrossover(thrust::device_vector<float>& newWeights, int weightPerModel, int parentPairNumber, curandState* state, unsigned blockNumber, unsigned threadNumber)
{
	int optBlockNumber = std::min(static_cast<unsigned>(parentPairNumber), threadNumber * blockNumber);
	crossoverSPCKernel << < optBlockNumber, threadNumber >> > (thrust::raw_pointer_cast(newWeights.data()), m_crossProbability, weightPerModel, parentPairNumber, state);
}