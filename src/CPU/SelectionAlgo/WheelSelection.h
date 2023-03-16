#pragma once
#include "GeneticSelector.h"

class WheelSelection : public GeneticSelector
{
public:
	WheelSelection()
	{
		m_floatDistr = std::uniform_real_distribution<double>(0.0, 1.0);
	}

	virtual void setSelector(const std::vector<std::pair<int, double>>& modelFitness) override
	{
		std::vector<double> fitnessValue;
		fitnessValue.reserve(modelFitness.size());
		std::transform(modelFitness.begin(), modelFitness.end(), std::back_inserter(fitnessValue), [](const auto& val) {return val.second; });
		m_cumulativeDistribution = std::discrete_distribution<>(fitnessValue.begin(), fitnessValue.end());
	}
	virtual std::pair<int, int> getParent(const std::vector<std::pair<int, double>>& modelFitness, std::mt19937& randEngine) override
	{
		int lhs = m_cumulativeDistribution(randEngine);
		int rhs = m_cumulativeDistribution(randEngine);
		while (lhs == rhs)
		{
			rhs = m_cumulativeDistribution(randEngine);
		}
		return { lhs,rhs };
	}
private:
	std::discrete_distribution<> m_cumulativeDistribution;
	std::uniform_real_distribution<double> m_floatDistr;
};
