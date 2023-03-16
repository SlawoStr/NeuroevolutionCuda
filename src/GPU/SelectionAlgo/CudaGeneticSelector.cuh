#pragma once
#include <thrust/device_vector.h>
#include "src/Utility/CudaRNG.cuh"

class CudaGeneticSelection
{
public:
	CudaGeneticSelection()
	{}
	void setParentNumber(unsigned parentPairNumber)
	{
		md_parent.resize(parentPairNumber * 2);
	}
	const thrust::device_vector<int>& getParents() { return md_parent; }

	virtual void runSelection(const std::vector<std::pair<int, double>>& modelFitness, curandState* state, unsigned blockNumber, unsigned threadNumber) = 0;

protected:
	thrust::device_vector<int> md_parent;
};


