#include "CudaNeuroevolution.cuh"
#include "src/Utility/CudaRNG.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>

__global__ void getParentWeight(float* inWeight, float* outWeight, int* parentIndex, int weightPerModel)
{
	for (int j = threadIdx.x; j < weightPerModel; j += blockDim.x)
	{
		outWeight[blockIdx.x * weightPerModel + j] = inWeight[parentIndex[blockIdx.x] * weightPerModel + j];
	}
}

CudaNeuroevolution::CudaNeuroevolution(CudaNeuralNetwork& network, unsigned parentPairNumber, unsigned blockNumber, unsigned threadNumber)
	: md_network(network), m_parentPairNumber(parentPairNumber), m_blockNumber(blockNumber), m_threadNumber(threadNumber)
{
	md_states = generateRandomStates(blockNumber, threadNumber, static_cast<unsigned>(time(0)));
	md_inWeights.resize(md_network.getNetworkWeightSize());
	md_outWeights.resize(md_inWeights.size());

	randomFloat(blockNumber, threadNumber, -1.0f, 1.0f, md_inWeights.size(), thrust::raw_pointer_cast(md_inWeights.data()), md_states);

	md_network.setWeight(thrust::raw_pointer_cast(md_inWeights.data()), md_inWeights.size(), false);
}

void CudaNeuroevolution::run(std::vector<std::pair<int, double>>&modelFitness)
{
	unsigned modelNumber = md_network.getModelNumber();
	unsigned weightPerModel = md_inWeights.size() / modelNumber;
	unsigned parentNumber = m_parentPairNumber * 2;
	// Selection
	m_selector->setParentNumber(m_parentPairNumber);
	m_selector->runSelection(modelFitness, md_states, m_blockNumber, m_threadNumber);
	auto parents = m_selector->getParents();
	// Elitism
	size_t survivors = modelNumber - parents.size();
	parents.resize(modelNumber);
	std::partial_sort(modelFitness.begin(), modelFitness.begin() + survivors, modelFitness.end(), [](const auto& lhs, const auto& rhs) {return lhs.second > rhs.second; });
	std::vector<int> bestParents(survivors);
	for (int j = 0; j < survivors; ++j)
	{
		bestParents[j] = modelFitness[j].first;
	}
	// Copy best parents to parent vector
	cudaMemcpy(thrust::raw_pointer_cast(parents.data() + parentNumber), bestParents.data(), sizeof(int) * survivors, cudaMemcpyHostToDevice);
	getParentWeight << <modelNumber, m_threadNumber >> >
		(thrust::raw_pointer_cast(md_inWeights.data()), thrust::raw_pointer_cast(md_outWeights.data()), thrust::raw_pointer_cast(parents.data()), weightPerModel);
	// Crossover
	m_crosser->runCrossover(md_outWeights, weightPerModel, m_parentPairNumber, md_states, m_blockNumber, m_threadNumber);
	// Mutation
	m_mutator->runMutation(md_outWeights, weightPerModel, parentNumber, md_states, m_blockNumber, m_threadNumber);
	// Weight transfer to neural network
	md_network.setWeight(thrust::raw_pointer_cast(md_outWeights.data()), md_outWeights.size(), false);
	md_outWeights.swap(md_inWeights);
}