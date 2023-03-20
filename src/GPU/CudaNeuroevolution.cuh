#pragma once
#include "CrossoverAlgo/CudaGeneticCrossover.cuh"
#include "SelectionAlgo/CudaGeneticSelector.cuh"
#include "MutationAlgo/CudaGeneticMutator.cuh"
#include "CudaNeuralNetwork.cuh"

class CudaNeuroevolution
{
public:
	CudaNeuroevolution(CudaNeuralNetwork& network, unsigned parentPairNumber, unsigned blockNumber = 15, unsigned threadNumber = 128);

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
	CudaNeuralNetwork md_network;
	curandState* md_states;
	unsigned m_parentPairNumber;
	// Genetic operators
	std::unique_ptr<CudaGeneticSelection> m_selector;
	std::unique_ptr<CudaGeneticCrossover> m_crosser;
	std::unique_ptr<CudaGeneticMutator> m_mutator;
	// Network weights array
	thrust::device_vector<float> md_inWeights;
	thrust::device_vector<float> md_outWeights;
	// Threading
	unsigned m_blockNumber;
	unsigned m_threadNumber;
};