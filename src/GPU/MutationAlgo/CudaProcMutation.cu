#include "CudaProcMutation.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void procMutationKernel(float* weight, float mutationProbability, float geneMutationProbability, int weightPerModel, int taskNumber, curandState* state, float minVal, float maxVal)
{
	__shared__ float mutationProba;
	curandState threadState = state[threadIdx.x + blockIdx.x * blockDim.x];
	for (int i = blockIdx.x; i < taskNumber; i += gridDim.x)
	{
		if (threadIdx.x == 0)
		{
			mutationProba = randomFloat(threadState, 0.0f, 1.0f);
		}
		__syncthreads();
		if (mutationProba < mutationProbability)
		{
			// Loop over genes
			for (int j = threadIdx.x; j < weightPerModel; j += blockDim.x)
			{
				if (randomFloat(threadState, 0.0f, 1.0f) < geneMutationProbability)
				{
					weight[i * weightPerModel + j] *= randomFloat(threadState, minVal, maxVal);
				}
			}
		}
		__syncthreads();
	}
	state[threadIdx.x + blockIdx.x * blockDim.x] = threadState;
}

void CudaProcMutation::runMutation(thrust::device_vector<float>& newWeights, int weightPerModel, int parentNumber, curandState* state, unsigned blockNumber, unsigned threadNumber)
{
	procMutationKernel << <std::min(blockNumber, static_cast<unsigned>(parentNumber)), threadNumber >> >
		(thrust::raw_pointer_cast(newWeights.data()), m_mutationProbability, m_geneMutationProbability, weightPerModel, parentNumber, state, m_minProc, m_maxProc);
}

