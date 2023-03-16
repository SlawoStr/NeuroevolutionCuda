#pragma once
#include <thrust/host_vector.h>
#include "CudaGeneticSelector.cuh"

class CudaWheelSelection : public CudaGeneticSelection
{
public:
	CudaWheelSelection(unsigned populationSize)
	{
		m_cumulativeDistribution.resize(populationSize);
		md_cumulativeDistribution.resize(populationSize);
	}
	void runSelection(const std::vector<std::pair<int, double>>& modelFitness, curandState* state, unsigned blockNumber, unsigned threadNumber)override;

private:
	thrust::host_vector<double> m_cumulativeDistribution;
	thrust::device_vector<double> md_cumulativeDistribution;
};