#pragma once
#include <vector>
#include <random>
#include "src/Utility/GeneticAlgorithmBadInput.h"


class GeneticMutator
{
public:
	GeneticMutator(float mutationProbability, float geneMutationProbability)
	{
		setMutationProbability(mutationProbability);
		setGeneMutationProbability(geneMutationProbability);
		m_floatDistr = std::uniform_real_distribution<float>(0.0f, 1.0f);
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
	virtual void runMutation(std::vector<float>& weight, std::mt19937& randEngine) = 0;
protected:
	float m_mutationProbability;
	float m_geneMutationProbability;
	std::uniform_real_distribution<float> m_floatDistr;
};