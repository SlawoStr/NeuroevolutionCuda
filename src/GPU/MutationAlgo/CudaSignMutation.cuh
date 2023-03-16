#pragma once
#include "CudaGeneticMutator.cuh"

class CudaSignMutation : public CudaGeneticMutator
{
public:
	CudaSignMutation(float mutationProbability, float geneMutationProbability) : CudaGeneticMutator(mutationProbability, geneMutationProbability)
	{}
	void runMutation(thrust::device_vector<float>& newWeights, int weightPerModel, int parentNumber, curandState* state, unsigned blockNumber, unsigned threadNumber) override;
};