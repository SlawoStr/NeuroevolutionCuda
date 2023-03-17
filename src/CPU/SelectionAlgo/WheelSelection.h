#pragma once
#include "GeneticSelector.h"

class WheelSelection : public GeneticSelector
{
public:
	WheelSelection(unsigned populationSize)
	{
		m_cumulativeDistribution.resize(populationSize);
	}

	int runWheel(int populationSize, std::mt19937& randEngine)
	{
		std::uniform_real_distribution<double> floatDistr{ 0.0f,1.0f };
		double randValue{ floatDistr(randEngine) };
		int winnerID = populationSize;
		for (int i = 1; i < populationSize; ++i)
		{
			if (randValue < m_cumulativeDistribution[i])
			{
				winnerID = i;
				break;
			}
		}
		return winnerID;
	}

	virtual void setSelector(const std::vector<std::pair<int, double>>& modelFitness) override
	{
		double fitnessSum{};
		for (const auto& value : modelFitness)
		{
			fitnessSum += value.second;
		}
		m_cumulativeDistribution[0] = modelFitness[0].second / fitnessSum;
		for (int i = 1; i < m_cumulativeDistribution.size(); ++i)
		{
			m_cumulativeDistribution[i] = m_cumulativeDistribution[i - 1] + modelFitness[i].second / fitnessSum;
		}
	}
	virtual std::pair<int, int> getParent(const std::vector<std::pair<int, double>>& modelFitness, std::mt19937& randEngine) override
	{
		int lhs = runWheel(static_cast<int>(modelFitness.size()), randEngine);
		int rhs = runWheel(static_cast<int>(modelFitness.size()), randEngine);
		while (lhs == rhs)
		{
			rhs = runWheel(static_cast<int>(modelFitness.size()), randEngine);
		}
		return { lhs,rhs };
	}
private:
	std::vector<double> m_cumulativeDistribution;
};
