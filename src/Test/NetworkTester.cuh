#pragma once
#include "src/CPU/NeuralNetwork.h"
#include "src/GPU/CudaNeuralNetwork.cuh"

class NetworkTester
{
public:
	NetworkTester(size_t inputSize, size_t outputSize, size_t modelNumber, size_t testCount, ActivationFunction outActFunc);

	void addLayer(size_t neuronNumber, ActivationFunction actFunc);

	void runTest();
private:
	void testWeightTransfer();
	void testPredictions();
	void testPredictionsSpeed();
private:
	std::vector<NeuralNetwork> m_cpuNetwork;
	CudaNeuralNetwork m_gpuNetwork;
	size_t m_inputSize;
	size_t m_outputSize;
	size_t m_modelNumber;
	size_t m_testNumber;
};

void startNetworkTest();
