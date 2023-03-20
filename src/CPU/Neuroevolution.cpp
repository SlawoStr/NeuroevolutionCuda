#include "Neuroevolution.h"
#include "src/Utility/RNGGenerator.h"
#include <omp.h>
#include <iostream>

std::vector<int> generateSwapParentIndex(std::vector<std::pair<int, double>>& modelFitness, unsigned parentNumber)
{
	size_t survivors = modelFitness.size() - parentNumber;
	std::partial_sort(modelFitness.begin(), modelFitness.begin() + survivors, modelFitness.end(), [](const auto& lhs, const auto& rhs) {return lhs.second > rhs.second; });
	std::vector<int> swapIndex;
	swapIndex.reserve(parentNumber);
	auto itStart = modelFitness.rbegin() + parentNumber;
	auto itEnd = modelFitness.rend();
	for (int i = 0; i < modelFitness.size(); ++i)
	{
		if (itStart != itEnd && itStart->first == i)
		{
			itStart++;
		}
		else
		{
			swapIndex.push_back(i);
		}
	}
	return swapIndex;
}

Neuroevolution::Neuroevolution(const NeuralNetwork& network, size_t modelNumber, unsigned parentPairNumber, unsigned threadNumber)
	: m_threadNumber{ threadNumber }, m_parentPairNumber{ parentPairNumber }
{
	// Generate random engines for each thread
	for (int i = 0; i < m_threadNumber; ++i)
	{
		m_randEngine.emplace_back(std::random_device{}());
	}
	// Generate neural networks
	for (int i = 0; i < modelNumber; ++i)
	{
		m_networks.push_back(network);
	}
	// Initialize networks with random weights
	std::vector<float> randWeights(network.getNetworkWeightSize());
	for (int i = 0; i < modelNumber; ++i)
	{
		RNGGenerator::randFloat(-1.0f, 1.0f, randWeights);
		m_weightIn.push_back(randWeights);
		m_networks[i].setWeight(randWeights);
	}
	m_weightOut = m_weightIn;
}

void Neuroevolution::run(std::vector<std::pair<int, double>>& modelFitness)
{
	m_selector->setSelector(modelFitness);
	#pragma omp parallel num_threads(m_threadNumber)
	{
		std::mt19937& threadEngine{ m_randEngine[omp_get_thread_num()] };
		std::vector<float> lhsWeight(m_weightIn[0].size());
		std::vector<float> rhsWeight(m_weightIn[0].size());
		#pragma omp for
		for (int i = 0; i < static_cast<int>(m_parentPairNumber); ++i)
		{
			// Select and get parent weights
			auto parents = m_selector->getParent(modelFitness, threadEngine);
			std::copy(m_weightIn[parents.first].begin(), m_weightIn[parents.first].end(), lhsWeight.begin());
			std::copy(m_weightIn[parents.second].begin(), m_weightIn[parents.second].end(), rhsWeight.begin());
			// Crossover
			m_crosser->runCrossover(lhsWeight, rhsWeight, threadEngine);
			// Mutation
			m_mutator->runMutation(lhsWeight, threadEngine);
			m_mutator->runMutation(rhsWeight, threadEngine);
			// Add new weights to global array
			std::copy(lhsWeight.begin(), lhsWeight.end(), m_weightOut[i * 2].begin());
			std::copy(rhsWeight.begin(), rhsWeight.end(), m_weightOut[i * 2 + 1].begin());
		}
	}
	std::swap(m_weightIn, m_weightOut);
	// Elite selection	
	auto swapIndex = generateSwapParentIndex(modelFitness, m_parentPairNumber * 2);
	for (int i = 0; i < swapIndex.size(); ++i)
	{
		m_networks[swapIndex[i]].setWeight(m_weightIn[i]);
	}
}

