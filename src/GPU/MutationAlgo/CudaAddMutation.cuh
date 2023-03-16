#pragma once
#include "CudaGeneticMutator.cuh"

class CudaAddMutation : public CudaGeneticMutator
{
public:
	CudaAddMutation(float mutationProbability, float geneMutationProbability, float addMin, float addMax) : CudaGeneticMutator(mutationProbability, geneMutationProbability)
	{
		setAddRange(addMin, addMax);
	}
	
	void setAddRange(float addMin, float addMax)
	{
		if (addMin >= addMax)
		{
			throw GeneticAlgorithmBadInput("Error: Minimum value to add cant be higher or equal to max value");
		}
		m_minAdd = addMin;
		m_maxAdd = addMax;
	}
	void runMutation(thrust::device_vector<float>& newWeights, int weightPerModel, int parentModel, curandState* state, unsigned blockNumber, unsigned threadNumber) override;

private:
	float m_minAdd;
	float m_maxAdd;
};