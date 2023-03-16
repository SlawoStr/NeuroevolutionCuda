#pragma once
#include "GeneticMutator.h"

class SignMuatation : public GeneticMutator
{
public:
	SignMuatation(float mutationProbability, float geneMutationProbability) : GeneticMutator(mutationProbability, geneMutationProbability)
	{}
	void runMutation(std::vector<float>& weight, std::mt19937& randEngine)
	{
		if (m_floatDistr(randEngine) < m_mutationProbability)
		{
			for (auto& val : weight)
			{
				if (m_floatDistr(randEngine) < m_geneMutationProbability)
				{
					val *= -1;
				}
			}
		}
	}
};