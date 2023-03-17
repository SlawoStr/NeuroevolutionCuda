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
		std::uniform_real_distribution<float> floatDistr{ 0.0f,1.0f };
		std::uniform_real_distribution<float> addDistr{ m_minAdd,m_maxAdd };
		if (floatDistr(randEngine) < m_mutationProbability)
		{
			for (auto& val : weight)
			{
				if (floatDistr(randEngine) < m_geneMutationProbability)
				{
					val += addDistr(randEngine);
				}
			}
		}
	}
private:
	float m_minAdd;
	float m_maxAdd;
};