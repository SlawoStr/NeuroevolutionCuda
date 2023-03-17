#pragma once
#include "GeneticMutator.h"

class NewMutation : public GeneticMutator
{
public:
	NewMutation(float mutationProbability, float geneMutationProbability) : GeneticMutator(mutationProbability, geneMutationProbability)
	{}
	void runMutation(std::vector<float>& weight, std::mt19937& randEngine)
	{
		std::uniform_real_distribution<float> floatDistr{ 0.0f,1.0f };
		std::uniform_real_distribution<float> newDistr{ -1.0f, 1.0f };
		if (floatDistr(randEngine) < m_mutationProbability)
		{
			for (auto& val : weight)
			{
				if (floatDistr(randEngine) < m_geneMutationProbability)
				{
					val = newDistr(randEngine);
				}
			}
		}
	}
};