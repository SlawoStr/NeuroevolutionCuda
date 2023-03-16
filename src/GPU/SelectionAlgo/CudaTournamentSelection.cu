#include "CudaTournamentSelection.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "src/Utility/GeneticAlgorithmBadInput.h"

__device__ int runTournament(double* fitness, int populationSize, int tournamentSize, curandState& state)
{
	int currentBest = randomInt(state, 0, populationSize);
	for (int i = 1; i < tournamentSize; ++i)
	{
		int newCompetitor = randomInt(state, 0, populationSize);
		if (fitness[newCompetitor] > fitness[currentBest])
		{
			currentBest = newCompetitor;
		}
	}
	return currentBest;
}

__global__ void tournamentSelectionKernel(double* fitness, int* parents, int parentNumber, int populationSize, int tournamentSize, curandState* state)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	curandState threadState = state[tid];
	for (int i = tid; i < parentNumber; i += blockDim.x * gridDim.x)
	{
		int lhs = runTournament(fitness, populationSize, tournamentSize, threadState);
		int rhs = runTournament(fitness, populationSize, tournamentSize, threadState);
		while (lhs == rhs)
		{
			rhs = runTournament(fitness, populationSize, tournamentSize, threadState);
		}
		parents[tid * 2] = lhs;
		parents[tid * 2 + 1] = rhs;
	}
	state[tid] = threadState;
}


void CudaTournamentSelection::setTournamentSize(unsigned populationSize, unsigned tournamentSize)
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

void CudaTournamentSelection::runSelection(const std::vector<std::pair<int, double>>& modelFitness, curandState* state, unsigned blockNumber, unsigned threadNumber)
{
	std::transform(modelFitness.begin(), modelFitness.end(), m_fitness.begin(), [](auto& val) {return val.second; });
	thrust::copy(m_fitness.begin(), m_fitness.end(), md_fitness.begin());

	int parentPair = static_cast<int>(md_parent.size() / 2);
	int optimalBlockNumber = std::min(blockNumber, (parentPair + threadNumber - 1) / threadNumber);
	tournamentSelectionKernel << <optimalBlockNumber, threadNumber >> > (thrust::raw_pointer_cast(md_fitness.data()), thrust::raw_pointer_cast(md_parent.data()), parentPair, static_cast<int>(md_fitness.size()), m_tournamentSize, state);
}