#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "CudaLayer.cuh"

/// <summary>
/// GPU Accelerated neural network for multi model predictions using cuda technology
/// </summary>
class CudaNeuralNetwork
{
public:
	CudaNeuralNetwork(size_t inputSize, size_t outputSize, size_t modelNumber, ActivationFunction outActFunc);

	void addLayer(size_t neuronNumber, ActivationFunction actFunc);

	void runPredictions(const float* ptr, size_t size, bool isHostAllocated, unsigned threadNumber);

	void getPredictions(float* ptr, size_t size, bool isHostAllocated);

	void getWeight(float* ptr, size_t size, bool isHostAllocated, unsigned threadNumber);

	void setWeight(const float* ptr, size_t size, bool isHostAllocated, unsigned threadNumber);

	size_t getNetworkWeightSize();

	size_t getModelNumber() const { return m_modelNumber; }
private:
	thrust::host_vector<CudaLayer> m_layers;
	size_t m_modelNumber;
};