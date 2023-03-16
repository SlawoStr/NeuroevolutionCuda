#pragma once
#include <thrust/host_vector.h>
#include "CudaGeneticSelector.cuh"

class CudaTruncationSelection : public CudaGeneticSelection
{
public:
	CudaTruncationSelection(unsigned populationSize, float bestParentRatio)
	{
		setBestParentRatio(populationSize, bestParentRatio);
	}

	void setBestParentRatio(unsigned populationSize, float bestParentRatio);

	void runSelection(const std::vector<std::pair<int, double>>& modelFitness, curandState* state, unsigned blockNumber, unsigned threadNumber)override;

private:
	thrust::host_vector<std::pair<int, double>> m_parentPool;
	thrust::host_vector<int> m_parentIndex;
	thrust::device_vector<int> md_parentIndex;
};