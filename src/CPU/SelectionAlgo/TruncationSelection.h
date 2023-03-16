#pragma once
#include "GeneticSelector.h"

class TruncationSelection : public GeneticSelector
{
public:
	TruncationSelection(unsigned populationSize, float bestParentRatio)
	{
		setBestParentRatio(populationSize, bestParentRatio);
	}

	void setSelector(const std::vector<std::pair<int, double>>& modelFitness) override
	{
		std::partial_sort_copy(modelFitness.begin(), modelFitness.end(), m_parentPool.begin(), m_parentPool.end(), [](const auto& lhs, const auto& rhs) {return lhs.second > rhs.second; });
	}


	std::pair<int, int> getParent(const std::vector<std::pair<int, double>>& modelFitness,std::mt19937& randEngine) override
	{
		int lhs = m_parentPool[m_intDistr(randEngine)].first;
		int rhs = m_parentPool[m_intDistr(randEngine)].first;
		while (lhs == rhs)
		{
			rhs = m_parentPool[m_intDistr(randEngine)].first;
		}
		return { lhs,rhs };
	}

	void setBestParentRatio(unsigned populationSize, float bestParentRatio)
	{
		if (bestParentRatio < 0.0f || bestParentRatio > 1.0f)
		{
			throw GeneticAlgorithmBadInput("Best parent ratio must be between 0.0f and 1.0f received value: " + std::to_string(bestParentRatio));
		}
		m_parentPool.resize(static_cast<size_t>(populationSize * bestParentRatio));
		// Reset int distibution range
		m_intDistr = std::uniform_int_distribution<int>(0, m_parentPool.size());
	}
private:
	std::vector<std::pair<int, double>> m_parentPool;
	std::uniform_int_distribution<int> m_intDistr;
};