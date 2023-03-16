#pragma once
#include "GeneticCrossover.h"

class UniformCrossover : public GeneticCrossover
{
public:
	UniformCrossover(float crossProbability) : GeneticCrossover(crossProbability)
	{}
	void runCrossover(std::vector<float>& lhs, std::vector<float>& rhs, std::mt19937& randEngine)
	{
		if (m_floatDistr(randEngine) < m_crossProbability)
		{
			for (int i = 0; i < lhs.size(); ++i)
			{
				if (m_floatDistr(randEngine) < 0.5f)
				{
					std::swap(lhs[i], rhs[i]);
				}
			}
		}
	}
	virtual void setCrossover(size_t weightNumber)
	{}
};