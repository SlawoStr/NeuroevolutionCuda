#pragma once
#include "CudaGeneticMutator.cuh"

class CudaProcMutation : public CudaGeneticMutator
{
public:
	CudaProcMutation(float mutationProbability, float geneMutationProbability, float procMin, float procMax) : CudaGeneticMutator(mutationProbability, geneMutationProbability)
	{
		setProcRange(procMin, procMax);
	}

	void setProcRange(float procMin, float procMax)
	{
		if (procMin >= procMax)
		{
			throw GeneticAlgorithmBadInput("Error: Minimum value to add cant be higher or equal to max value");
		}
		m_minProc = procMin;
		m_maxProc = procMax;
	}
	void runMutation(thrust::device_vector<float>& newWeights, int weightPerModel, int parentModel, curandState* state, unsigned blockNumber, unsigned threadNumber) override;

private:
	float m_minProc;
	float m_maxProc;
};