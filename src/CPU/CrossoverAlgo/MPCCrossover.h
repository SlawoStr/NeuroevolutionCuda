#pragma once
#include "GeneticCrossover.h"

class MPCCrossover : public GeneticCrossover
{
public:
	MPCCrossover(float crossProbability) : GeneticCrossover(crossProbability)
	{
	}
	void runCrossover(std::vector<float>& lhs, std::vector<float>& rhs, std::mt19937& randEngine)
	{
		if (m_floatDistr(randEngine) < m_crossProbability)
		{
			int lhsPoint = m_intDistr(randEngine);
			int rhsPoint = m_intDistr(randEngine);
			while (lhsPoint == rhsPoint)
			{
				rhsPoint = m_intDistr(randEngine);
			}
			if (lhsPoint > rhsPoint)
			{
				std::swap(lhsPoint, rhsPoint);
			}
			std::swap_ranges(lhs.begin() + lhsPoint, lhs.begin() + rhsPoint, rhs.begin() + lhsPoint);
		}
	}
	virtual void setCrossover(size_t weightNumber)
	{
		m_intDistr = std::uniform_int_distribution<int>(1, weightNumber - 1);
	}
private:
	std::uniform_int_distribution<int> m_intDistr;
};
