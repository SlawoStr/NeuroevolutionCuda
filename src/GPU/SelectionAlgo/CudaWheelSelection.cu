#include "CudaWheelSelection.cuh"
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ int runWheel(double* cmdf, int populationSize, curandState& state)
{
	float randomValue{ randomFloat(state,0.0f,1.0f) };
	int winnerID{ populationSize - 1 };
	for (int i = 0; i < populationSize; ++i)
	{
		if (randomValue < cmdf[i])
		{
			winnerID = i;
			break;
		}
	}
	return winnerID;
}

__global__ void wheelSelectionKernel(double* cumulativeDistribution, int* parents, int populationSize, int parentNumber, curandState* state)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	curandState threadState = state[tid];
	for (int i = tid; i < parentNumber; i += blockDim.x * gridDim.x)
	{
		int lhs = runWheel(cumulativeDistribution, populationSize, threadState);
		int rhs = runWheel(cumulativeDistribution, populationSize, threadState);
		while (lhs == rhs)
		{
			rhs = runWheel(cumulativeDistribution, populationSize, threadState);
		}
		parents[tid * 2] = lhs;
		parents[tid * 2 + 1] = rhs;
	}
	state[tid] = threadState;
}

void CudaWheelSelection::runSelection(const std::vector<std::pair<int, double>>& modelFitness, curandState* state, unsigned blockNumber, unsigned threadNumber)
{
	double fitnessSum{};
	for (const auto& value : modelFitness)
	{
		fitnessSum += value.second;
	}
	m_cumulativeDistribution[0] = modelFitness[0].second / fitnessSum;
	for (int i = 1; i < m_cumulativeDistribution.size(); ++i)
	{
		m_cumulativeDistribution[i] = m_cumulativeDistribution[i - 1] + modelFitness[i].second / fitnessSum;
	}
	thrust::copy(m_cumulativeDistribution.begin(), m_cumulativeDistribution.end(), md_cumulativeDistribution.begin());
}