#pragma once
#include <thrust/host_vector.h>
#include <algorithm>
#include "CudaGeneticSelector.cuh"

class CudaTournamentSelection : public CudaGeneticSelection
{
public:
	CudaTournamentSelection(unsigned populationSize, unsigned tournamentSize)
	{
		setTournamentSize(populationSize, tournamentSize);
		m_fitness.resize(populationSize);
		md_fitness.resize(populationSize);
	}
	void setTournamentSize(unsigned populationSize, unsigned tournamentSize);
	void runSelection(const std::vector<std::pair<int, double>>& modelFitness, curandState* state, unsigned blockNumber, unsigned threadNumber)override;
private:
	unsigned m_tournamentSize;
	thrust::host_vector<double> m_fitness;
	thrust::device_vector<double> md_fitness;
};