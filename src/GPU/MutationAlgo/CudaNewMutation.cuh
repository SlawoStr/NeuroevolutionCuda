#pragma once
#include "CudaGeneticMutator.cuh"

class CudaNewMutation : public CudaGeneticMutator
{
public:
	CudaNewMutation(float mutationProbability, float geneMutationProbability) : CudaGeneticMutator(mutationProbability, geneMutationProbability)
	{}
	void runMutation(thrust::device_vector<float>& newWeights, int weightPerModel, int parentModel, curandState* state, unsigned blockNumber, unsigned threadNumber) override;
};