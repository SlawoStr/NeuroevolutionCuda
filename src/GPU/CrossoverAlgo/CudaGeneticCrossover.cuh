#pragma once
#include "src/Utility/GeneticAlgorithmBadInput.h"
#include <thrust/device_vector.h>
#include "src/Utility/CudaRNG.cuh"

class CudaGeneticCrossover
{
public:
	CudaGeneticCrossover(float crossProbability)
	{
		setCrossoverProbability(crossProbability);
	}

	void setCrossoverProbability(float crossProbability)
	{
		if (crossProbability > 1.0f || crossProbability < 0.0f)
		{
			throw GeneticAlgorithmBadInput("Error: Crossover probability must be in range[0,1] received value: " + std::to_string(crossProbability));
		}
		m_crossProbability = crossProbability;
	}
	virtual void runCrossover(thrust::device_vector<float>& newWeights, int weightPerModel, int parentPairNumber, curandState* state, unsigned blockNumber, unsigned threadNumber) = 0;

protected:
	float m_crossProbability;
};