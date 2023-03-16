#pragma once
#include "GeneticMutator.h"

class AddMutation : public GeneticMutator
{
public:
	AddMutation(float mutationProbability, float geneMutationProbability, float addMin, float addMax)
		: GeneticMutator(mutationProbability, geneMutationProbability)
	{
		setAddRange(addMin, addMax);
	}

	void setAddRange(float addMin, float addMax)
	{
		if (addMin >= addMax)
		{
			throw GeneticAlgorithmBadInput("Error: Minimum value to add cant be higher or equal to max value");
		}
		m_minAdd = addMin;
		m_maxAdd = addMax;
	}

	void runMutation(std::vector<float>& weight, std::mt19937& randEngine)
	{
		if (m_floatDistr(randEngine) < m_mutationProbability)
		{
			for (auto& val : weight)
			{
				if (m_floatDistr(randEngine) < m_geneMutationProbability)
				{
					val += m_addDistr(randEngine);
				}
			}
		}
	}
private:
	float m_minAdd;
	float m_maxAdd;
	std::uniform_real_distribution<float> m_addDistr;
};