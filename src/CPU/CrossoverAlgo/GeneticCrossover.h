#pragma once
#include <vector>
#include <random>
#include "src/Utility/GeneticAlgorithmBadInput.h"

class GeneticCrossover
{
public:
	GeneticCrossover(float crossProbability)
	{
		setCrossoverProbability(crossProbability);
		m_floatDistr = std::uniform_real_distribution<float>(0.0f, 1.0f);
	}
	void setCrossoverProbability(float crossProbability)
	{
		if (crossProbability > 1.0f || crossProbability < 0.0f)
		{
			throw GeneticAlgorithmBadInput("Error: Crossover probability must be in range[0,1] received value: " + std::to_string(crossProbability));
		}
		m_crossProbability = crossProbability;
	}
	virtual void runCrossover(std::vector<float>& lhs, std::vector<float>& rhs, std::mt19937& rangEngine) = 0;

	virtual void setCrossover(size_t weightNumber) = 0;
protected:
	float m_crossProbability;
	std::uniform_real_distribution<float> m_floatDistr;
};