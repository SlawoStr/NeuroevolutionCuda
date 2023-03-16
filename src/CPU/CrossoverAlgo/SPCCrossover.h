#pragma once
#include "GeneticCrossover.h"

class SPCCrossover : public GeneticCrossover
{
public:
	SPCCrossover(float crossProbability) : GeneticCrossover(crossProbability)
	{}

	void runCrossover(std::vector<float>& lhs, std::vector<float>& rhs, std::mt19937& randEngine)
	{
		if (m_floatDistr(randEngine) < m_crossProbability)
		{
			int dividePoint = m_intDistr(randEngine);
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

	virtual void setCrossover(size_t weightNumber)
	{
		m_intDistr = std::uniform_int_distribution<int>(1, weightNumber - 1);
	}

private:
	std::uniform_int_distribution<int> m_intDistr;
};