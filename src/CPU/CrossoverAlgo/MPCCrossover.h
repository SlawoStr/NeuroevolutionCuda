#pragma once
#include "GeneticCrossover.h"

class MPCCrossover : public GeneticCrossover
{
public:
	MPCCrossover(float crossProbability) : GeneticCrossover(crossProbability)
	{}
	void runCrossover(std::vector<float>& lhs, std::vector<float>& rhs, std::mt19937& randEngine)
	{
		std::uniform_real_distribution<float> floatDistr{ 0.0f,1.0f };
		std::uniform_int_distribution<int> intDistr{ 1,static_cast<int>(rhs.size() - 2) };

		if (floatDistr(randEngine) < m_crossProbability)
		{
			int lhsPoint = intDistr(randEngine);
			int rhsPoint = intDistr(randEngine);
			while (lhsPoint == rhsPoint)
			{
				rhsPoint = intDistr(randEngine);
			}
			if (lhsPoint > rhsPoint)
			{
				std::swap(lhsPoint, rhsPoint);
			}
			std::swap_ranges(lhs.begin() + lhsPoint, lhs.begin() + rhsPoint, rhs.begin() + lhsPoint);
		}
	}
};
