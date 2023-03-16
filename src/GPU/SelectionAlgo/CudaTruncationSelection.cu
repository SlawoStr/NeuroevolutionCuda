#include "CudaTruncationSelection.cuh"
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "src/Utility/GeneticAlgorithmBadInput.h"


__global__ void truncationSelectionKernel(double* fitness, int* parents, int* parentIndex, int parentNumber, int populationSize, int bestParentNumber, curandState* state)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	curandState threadState = state[tid];
	for (int i = tid; i < parentNumber; i += blockDim.x * gridDim.x)
	{
		int lhs = randomInt(threadState, 0, bestParentNumber);
		int rhs = randomInt(threadState, 0, bestParentNumber);
		while (lhs == rhs)
		{
			rhs = randomInt(threadState, 0, bestParentNumber);
		}
		parents[tid * 2] = parentIndex[lhs];
		parents[tid * 2 + 1] = parentIndex[rhs];
	}
	state[tid] = threadState;
}

void CudaTruncationSelection::setBestParentRatio(unsigned populationSize, float bestParentRatio)
{
	if (bestParentRatio < 0.0f || bestParentRatio > 1.0f)
	{
		throw GeneticAlgorithmBadInput("Best parent ratio must be between 0.0f and 1.0f received value: " + std::to_string(bestParentRatio));
	}
	size_t bestParentNumber = static_cast<size_t>(populationSize * bestParentRatio);
	m_parentPool.resize(bestParentNumber);
	m_parentIndex.resize(bestParentNumber);
	md_parentIndex.resize(bestParentNumber);
}

void CudaTruncationSelection::runSelection(const std::vector<std::pair<int, double>>& modelFitness, curandState* state, unsigned blockNumber, unsigned threadNumber)
{
	std::partial_sort_copy(modelFitness.begin(), modelFitness.end(), m_parentPool.begin(), m_parentPool.end(), [](const auto& lhs, const auto& rhs) {return lhs.second > rhs.second; });
	std::transform(m_parentPool.begin(), m_parentPool.end(), m_parentIndex.begin(), [](auto& val) {return val.first; });
	thrust::copy(m_parentIndex.begin(), m_parentIndex.end(), md_parentIndex.begin());
}