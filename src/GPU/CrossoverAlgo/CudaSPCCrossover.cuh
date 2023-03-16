#pragma once
#include "CudaGeneticCrossover.cuh"

class CudaSPCCrossover : public CudaGeneticCrossover
{
public:
	CudaSPCCrossover(float crossoverProbability) : CudaGeneticCrossover(crossoverProbability)
	{}

	void runCrossover(thrust::device_vector<float>& newWeights, int weightPerModel, int parentPairNumber, curandState* state, unsigned blockNumber, unsigned threadNumber)override;
};