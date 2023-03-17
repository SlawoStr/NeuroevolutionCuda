#pragma once
#include "GeneticSelector.h"

class TournamentSelection : public GeneticSelector
{
public:
	TournamentSelection(unsigned populationSize, unsigned tournamentSize)
	{
		setTournamentSize(populationSize, tournamentSize);
	}

	void setTournamentSize(unsigned populationSize, unsigned tournamentSize)
	{
		if (tournamentSize >= populationSize)
		{
			throw GeneticAlgorithmBadInput("Tournament size must be smaller than population size MaxSize: "
				+ std::to_string(populationSize) + " received: " + std::to_string(tournamentSize));
		}
		else if (tournamentSize <= 0)
		{
			throw GeneticAlgorithmBadInput("Tournament size must be bigger than population size MinSize: 0 received: " + std::to_string(tournamentSize));
		}
		m_tournamentSize = tournamentSize;
	}

	virtual void setSelector(const std::vector<std::pair<int, double>>& modelFitness) override
	{}

	virtual std::pair<int, int> getParent(const std::vector<std::pair<int, double>>& modelFitness,std::mt19937& randEngine) override
	{
		int lhs = runTournament(modelFitness, randEngine);
		int rhs = runTournament(modelFitness, randEngine);
		while (lhs == rhs)
		{
			rhs = runTournament(modelFitness, randEngine);
		}
		return { lhs,rhs };
	}

private:
	unsigned runTournament(const std::vector<std::pair<int, double>>& modelFitness, std::mt19937& randEngine)
	{
		std::uniform_int_distribution<int> intDistr{ 0,static_cast<int>(modelFitness.size()) - 1 };
		int currentBest = intDistr(randEngine);
		for (unsigned i = 1; i < m_tournamentSize; ++i)
		{
			int newCompetitor = intDistr(randEngine);
			if (modelFitness[newCompetitor].second > modelFitness[currentBest].second)
			{
				currentBest = newCompetitor;
			}
		}
		return currentBest;
	}
private:
	unsigned m_tournamentSize;
};