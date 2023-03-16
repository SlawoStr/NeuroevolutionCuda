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
		m_procAdd = procMin;
		m_procAdd = procMax;
	}
	void runMutation(std::vector<float>& weight, std::mt19937& randEngine)
	{
		if (m_floatDistr(randEngine) < m_mutationProbability)
		{
			for (auto& val : weight)
			{
				if (m_floatDistr(randEngine) < m_geneMutationProbability)
				{
					val *= m_procDistr(randEngine);
				}
			}
		}
	}
private:
	float m_procAdd;
	float m_procAdd;
	std::uniform_real_distribution<float> m_procDistr;
};