#include "NetworkTester.cuh"
#include "src/Utility/RNGGenerator.h"
#include "src/Utility/Timer.h"
#include <numeric>

NetworkTester::NetworkTester(size_t inputSize, size_t outputSize, size_t modelNumber, size_t testCount, ActivationFunction outActFunc)
	: m_cpuNetwork(inputSize, outputSize, outActFunc), m_gpuNetwork(inputSize, outputSize, modelNumber, outActFunc), m_inputSize{ inputSize }, m_outputSize{ outputSize }, m_testNumber{ testCount },m_modelNumber{modelNumber}
{
}

void NetworkTester::addLayer(size_t neuronNumber, ActivationFunction actFunc)
{
	m_cpuNetwork.addLayer(neuronNumber, actFunc);
	m_gpuNetwork.addLayer(neuronNumber, actFunc);
}

void NetworkTester::runTest()
{
	testWeightTransfer();
	testPredictions();
	testPredictionsSpeed();
}

void NetworkTester::testWeightTransfer()
{
	size_t cpuNetSize = m_cpuNetwork.getNetworkWeightSize();
	size_t gpuNetSize = m_gpuNetwork.getNetworkWeightSize();

	if (cpuNetSize != gpuNetSize / m_modelNumber)
	{
		std::cout << "ERROR: Size of gpu and cpu network are not equal" << std::endl;
	}
	else
	{
		std::cout << "Weight size passed" << std::endl;
	}
	std::vector<float> cpuNetworkWeight(cpuNetSize);
	std::vector<float> gpuNetworkWeight(gpuNetSize);

	for (int i = 0; i < cpuNetworkWeight.size(); ++i)
	{
		cpuNetworkWeight[i] = RNGGenerator::randFloat(-1.0f, 1.0f);
	}

	for (int i = 0; i < m_modelNumber; ++i)
	{
		for (int j = 0; j < cpuNetworkWeight.size(); ++j)
		{
			gpuNetworkWeight[i * cpuNetworkWeight.size() + j] = cpuNetworkWeight[j];
		}
	}

	m_cpuNetwork.setWeight(cpuNetworkWeight);
	m_gpuNetwork.setWeight(gpuNetworkWeight.data(), gpuNetworkWeight.size(), true, 128);
	gpuNetworkWeight.clear();
	gpuNetworkWeight.resize(m_gpuNetwork.getNetworkWeightSize());
	m_gpuNetwork.getWeight(gpuNetworkWeight.data(), gpuNetworkWeight.size(), true, 128);

	bool isCorrect{ true };
	for (int i = 0; i < m_modelNumber; ++i)
	{
		for (int j = 0; j < cpuNetworkWeight.size(); ++j)
		{
			if (fabs(gpuNetworkWeight[i * cpuNetworkWeight.size() + j] - cpuNetworkWeight[j]) < DBL_EPSILON)
			{
				continue;
			}
			else
			{
				isCorrect = false;
			}
		}
	}
	if (!isCorrect)
	{
		std::cout << "Error: Weight transfer is incorrect" << std::endl;
	}
	else
	{
		std::cout << "Weights transfer is correct" << std::endl;
	}
}

void NetworkTester::testPredictions()
{
	std::vector<float> testWeights(m_gpuNetwork.getNetworkWeightSize());
	std::vector<float> testInput(m_inputSize * m_modelNumber);

	for (auto& weight : testWeights)
	{
		weight = RNGGenerator::randFloat(-1.0f, 1.0f);
	}
	for (auto& input : testInput)
	{
		input = RNGGenerator::randFloat(-1.0f, 1.0f);
	}

	m_gpuNetwork.setWeight(testWeights.data(), testWeights.size(), true, 128);
	m_gpuNetwork.runPredictions(testInput.data(), testInput.size(), true, 128);

	std::vector<float> gpuResult(m_outputSize * m_modelNumber);
	std::vector<float> cpuResult(m_outputSize * m_modelNumber);
	m_gpuNetwork.getPredictions(gpuResult.data(), gpuResult.size(), true);

	std::vector<float> cpuWeight(m_cpuNetwork.getNetworkWeightSize());
	std::vector<float> cpuInput(m_inputSize);

	for (int i = 0; i < m_modelNumber; ++i)
	{
		std::copy(testWeights.begin() + i * cpuWeight.size(), testWeights.begin() + (i + 1) * cpuWeight.size(), cpuWeight.begin());
		std::copy(testInput.begin() + i * cpuInput.size(), testInput.begin() + (i + 1) * cpuInput.size(), cpuInput.begin());
		m_cpuNetwork.setWeight(cpuWeight);
		auto res = m_cpuNetwork.predict(cpuInput);
		for (int j = 0; j < res.size(); ++j)
		{
			cpuResult[i * m_outputSize + j] = res[j];
		}
	}

	int correctValues{};
	int incorrectValues{};
	for (int i = 0; i < gpuResult.size(); ++i)
	{
		if (fabs(gpuResult[i] - cpuResult[i]) < 0.0001)
		{
			correctValues++;
		}
		else
		{
			std::cout << "ID: " << i << "\tGPU: " << gpuResult[i] << "\tCPU : " << cpuResult[i] << std::endl;
			incorrectValues++;
		}
	}

	std::cout << "Prediction finished correct values: " << correctValues << " Incorrect values: " << incorrectValues << std::endl;
}

void NetworkTester::testPredictionsSpeed()
{
	// Time
	Timer t;
	double gpuTime{};
	double gpuTimeFast{};
	double cpuTime{};
	double resultDiff{};
	// Set up data
	std::vector<float> testWeights(m_gpuNetwork.getNetworkWeightSize());
	std::vector<float> testInput(m_inputSize * m_modelNumber);
	thrust::device_vector<float> gpuTestInput(m_inputSize * m_modelNumber);
	std::vector<float> cpuWeight(m_cpuNetwork.getNetworkWeightSize());
	std::vector<float> cpuInput(m_inputSize);
	std::vector<float> gpuResultVec(m_outputSize * m_modelNumber);
	thrust::device_vector<float> gpuResultVecFast(m_outputSize * m_modelNumber);

	for (unsigned i = 0; i < m_testNumber; ++i)
	{
		std::cout << "Startin test: " << i << std::endl;
		// Results
		double cpuResult{};
		double gpuResult{};
		double gpuResultFast{};
		// Init data
		for (auto& weight : testWeights)
		{
			weight = RNGGenerator::randFloat(-1.0f, 1.0f);
		}
		for (auto& input : testInput)
		{
			input = RNGGenerator::randFloat(-1.0f, 1.0f);
		}
		cudaMemcpy(thrust::raw_pointer_cast(gpuTestInput.data()), testInput.data(), sizeof(float) * testInput.size(), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		for (int i = 0; i < m_modelNumber; ++i)
		{
			std::copy(testWeights.begin() + i * cpuWeight.size(), testWeights.begin() + (i + 1) * cpuWeight.size(), cpuWeight.begin());
			std::copy(testInput.begin() + i * cpuInput.size(), testInput.begin() + (i + 1) * cpuInput.size(), cpuInput.begin());
			m_cpuNetwork.setWeight(cpuWeight);
			t.start();
			auto res = m_cpuNetwork.predict(cpuInput);
			t.stop();
			cpuTime += t.measure();
			cpuResult += std::accumulate(res.begin(), res.end(), 0.0f);
		}
		m_gpuNetwork.setWeight(testWeights.data(), testWeights.size(), true, 128);
		cudaDeviceSynchronize();
		t.start();
		m_gpuNetwork.runPredictions(testInput.data(), testInput.size(), true, 64);
		m_gpuNetwork.getPredictions(gpuResultVec.data(), gpuResultVec.size(), true);
		t.stop();
		gpuTime += t.measure();
		gpuResult = std::accumulate(gpuResultVec.begin(), gpuResultVec.end(), 0.0f);
		cudaDeviceSynchronize();
		t.start();
		m_gpuNetwork.runPredictions(thrust::raw_pointer_cast(gpuTestInput.data()), testInput.size(), false, 64);
		m_gpuNetwork.getPredictions(thrust::raw_pointer_cast(gpuResultVecFast.data()), gpuResultVec.size(), false);
		cudaDeviceSynchronize();
		t.stop();
		gpuTimeFast += t.measure();
		gpuResultFast = thrust::reduce(gpuResultVecFast.begin(), gpuResultVecFast.end(), 0.0f);
		if (fabs(gpuResult - gpuResultFast) < 0.0001)
		{
			std::cout << "Gpu results are equal" << std::endl;
		}
		resultDiff += fabs(gpuResult - cpuResult);
	}
	std::cout << "Time results" << std::endl;
	std::cout << "CPU TIME: " << cpuTime * 1000 / m_testNumber << std::endl;
	std::cout << "GPU TIME: " << gpuTime * 1000 / m_testNumber << std::endl;
	std::cout << "GPU TIME (WITHOUT TRANSFER)" << gpuTimeFast* 1000 / m_testNumber << std::endl;
	std::cout << "Prediction difference: " << resultDiff / m_testNumber << std::endl;
}



void startNetworkTest()
{
	{
		size_t inputSize{ 250 };
		size_t outputSize{ 4 };
		size_t modelNumber{ 100000 };
		size_t testCount{ 10 };
		ActivationFunction outActFunc = ActivationFunction::SIGMOID;

		NetworkTester tester(inputSize, outputSize, modelNumber, testCount, outActFunc);

		tester.addLayer(20, ActivationFunction::RELU);
		tester.addLayer(10, ActivationFunction::RELU);
		tester.runTest();
	}
	/*
	{
		size_t inputSize{ 3500 };
		size_t outputSize{ 100 };
		size_t modelNumber{ 50 };
		size_t testCount{ 1 };
		ActivationFunction outActFunc = ActivationFunction::RELU;

		NetworkTester tester(inputSize, outputSize, modelNumber, testCount, outActFunc);

		//tester.addLayer(100, ActivationFunction::RELU);
		//tester.addLayer(50, ActivationFunction::RELU);
		tester.runTest();
	}
	*/

}