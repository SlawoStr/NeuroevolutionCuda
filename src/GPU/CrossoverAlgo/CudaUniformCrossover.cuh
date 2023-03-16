#pragma once
#include "CudaGeneticCrossover.cuh"

class CudaUniformCrossover : public CudaGeneticCrossover
{
public:
	CudaUniformCrossover(float crossoverProbability) : CudaGeneticCrossover(crossoverProbability)
	{}

	void runCrossover(thrust::device_vector<float>& newWeights, int weightPerModel, int parentPairNumber, curandState* state, unsigned blockNumber, unsigned threadNumber)override;
};