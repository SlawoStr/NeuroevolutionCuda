#pragma once
#include "SelectionAlgo/GeneticSelector.h"
#include "CrossoverAlgo/GeneticCrossover.h"
#include "MutationAlgo/GeneticMutator.h"
#include "NeuralNetwork.h"
#include <memory>
#include <vector>
#include <random>

class Neuroevolution
{
public:
	Neuroevolution(const NeuralNetwork& network, size_t modelNumber, unsigned parentPairNumber, unsigned threadNumber);

	void run(std::vector<std::pair<int, double>>& modelFitness);

	template<typename C, typename... Args>
	void setSelection(Args... args)
	{
		m_selector = std::make_unique<C>(args...);
	}
	template<typename C, typename... Args>
	void setCrossover(Args... args)
	{
		m_crosser = std::make_unique<C>(args...);
	}
	template<typename C, typename... Args>
	void setMutation(Args... args)
	{
		m_mutator = std::make_unique<C>(args...);
	}
private:
	// Genetic operators
	std::unique_ptr<GeneticSelector> m_selector;
	std::unique_ptr<GeneticCrossover> m_crosser;
	std::unique_ptr<GeneticMutator> m_mutator;
	// Parameters
	unsigned m_parentPairNumber;
	unsigned m_threadNumber;
	std::vector<std::mt19937> m_randEngine;
	std::vector<NeuralNetwork> m_networks;
	// Network weight containters
	std::vector<std::vector<float>> m_weightIn;
	std::vector<std::vector<float>> m_weightOut;
};