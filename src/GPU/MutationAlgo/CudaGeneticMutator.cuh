#pragma once
#include "src/Utility/GeneticAlgorithmBadInput.h"
#include <thrust/device_vector.h>
#include "src/Utility/CudaRNG.cuh"

class CudaGeneticMutator
{
public:
	CudaGeneticMutator(float mutationProbability, float geneMutationProbability)
	{
		setMutationProbability(mutationProbability);
		setGeneMutationProbability(geneMutationProbability);
	}

	void setMutationProbability(float mutationProbability)
	{
		if (mutationProbability < 0.0f || mutationProbability > 1.0f)
		{
			throw GeneticAlgorithmBadInput("Error: Mutation probability must be in range [0,1] received value: " + std::to_string(mutationProbability));
		}
		m_mutationProbability = mutationProbability;
	}

	void setGeneMutationProbability(float geneMutationProbability)
	{
		if (geneMutationProbability <= 0.0f || geneMutationProbability > 1.0f)
		{
			throw GeneticAlgorithmBadInput("Error: Gene mutation probability must be in range (0,1] received value: " + std::to_string(geneMutationProbability));
		}
		m_geneMutationProbability = geneMutationProbability;
	}

	virtual void runMutation(thrust::device_vector<float>& newWeights, int weightPerModel, int parentModel, curandState* state, unsigned blockNumber, unsigned threadNumber) = 0;


protected:
	float m_mutationProbability;
	float m_geneMutationProbability;
};