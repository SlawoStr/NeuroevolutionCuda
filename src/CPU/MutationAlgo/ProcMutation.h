#pragma once
#include "GeneticMutator.h"

class ProcMutation : public GeneticMutator
{
public:
	ProcMutation(float mutationProbability, float geneMutationProbability, float procMin, float procMax)
		: GeneticMutator(mutationProbability, geneMutationProbability)
	{
		setProcRange(procMin, procMax);
	}
	void setProcRange(float procMin, float procMax)
	{
		if (procMin >= procMax)
		{
			throw GeneticAlgorithmBadInput("Error: Minimum value to add cant be higher or equal to max value");
		}
		m_procMin = procMin;
		m_procMax = procMax;
	}
	void runMutation(std::vector<float>& weight, std::mt19937& randEngine)
	{
		std::uniform_real_distribution<float> floatDistr{ 0.0f,1.0f };
		std::uniform_real_distribution<float> procDistr{ m_procMin,m_procMax };
		if (floatDistr(randEngine) < m_mutationProbability)
		{
			for (auto& val : weight)
			{
				if (floatDistr(randEngine) < m_geneMutationProbability)
				{
					val *= procDistr(randEngine);
				}
			}
		}
	}
private:
	float m_procMin;
	float m_procMax;
};