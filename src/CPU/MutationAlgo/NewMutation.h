#pragma once
#include "GeneticMutator.h"

class NewMutation : public GeneticMutator
{
public:
	NewMutation(float mutationProbability, float geneMutationProbability) : GeneticMutator(mutationProbability, geneMutationProbability)
	{
		m_newDistr = std::uniform_real_distribution<float>(-1.0f, 1.0f);
	}

	void runMutation(std::vector<float>& weight, std::mt19937& randEngine)
	{
		if (m_floatDistr(randEngine) < m_mutationProbability)
		{
			for (auto& val : weight)
			{
				if (m_floatDistr(randEngine) < m_geneMutationProbability)
				{
					val = m_newDistr(randEngine);
				}
			}
		}
	}
private:
	std::uniform_real_distribution<float> m_newDistr;
};