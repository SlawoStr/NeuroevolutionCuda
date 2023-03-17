#pragma once
#include "GeneticCrossover.h"

class SPCCrossover : public GeneticCrossover
{
public:
	SPCCrossover(float crossProbability) : GeneticCrossover(crossProbability)
	{}
	void runCrossover(std::vector<float>& lhs, std::vector<float>& rhs, std::mt19937& randEngine)
	{
		std::uniform_real_distribution<float> floatDistr{ 0.0f,1.0f };
		std::uniform_int_distribution<int> intDistr{ 1,static_cast<int>(rhs.size() - 2) };
		if (floatDistr(randEngine) < m_crossProbability)
		{
			int dividePoint = intDistr(randEngine);
			if (dividePoint < lhs.size() / 2)
			{
				std::swap_ranges(lhs.begin(), lhs.begin() + dividePoint, rhs.begin());
			}
			else
			{
				std::swap_ranges(lhs.begin() + dividePoint, lhs.end(), rhs.begin() + dividePoint);
			}
		}
	}
};