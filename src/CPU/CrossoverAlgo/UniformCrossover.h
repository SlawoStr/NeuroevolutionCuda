#pragma once
#include "GeneticCrossover.h"

class UniformCrossover : public GeneticCrossover
{
public:
	UniformCrossover(float crossProbability) : GeneticCrossover(crossProbability)
	{}
	void runCrossover(std::vector<float>& lhs, std::vector<float>& rhs, std::mt19937& randEngine)
	{
		std::uniform_real_distribution<float> floatDistr{ 0.0f,1.0f };
		if (floatDistr(randEngine) < m_crossProbability)
		{
			for (int i = 0; i < lhs.size(); ++i)
			{
				if (floatDistr(randEngine) < 0.5f)
				{
					std::swap(lhs[i], rhs[i]);
				}
			}
		}
	}
};