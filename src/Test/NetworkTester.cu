#include "NetworkTester.cuh"
#include "src/Utility/RNGGenerator.h"
#include "src/Utility/Timer.h"
#include <numeric>

NetworkTester::NetworkTester(size_t inputSize, size_t outputSize, size_t modelNumber, size_t testCount, ActivationFunction outActFunc)
	: m_gpuNetwork(inputSize, outputSize, modelNumber, outActFunc), m_inputSize{ inputSize }, m_outputSize{ outputSize }, m_testNumber{ testCount },m_modelNumber{modelNumber}
{
	for (int i = 0; i < m_modelNumber; ++i)
	{
		m_cpuNetwork.emplace_back(inputSize, outputSize, outActFunc);
	}
}

void NetworkTester::addLayer(size_t neuronNumber, ActivationFunction actFunc)
{
	for (int i = 0; i < m_modelNumber; ++i)
	{
		m_cpuNetwork[i].addLayer(neuronNumber, actFunc);

	}
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
	size_t cpuNetSize = m_cpuNetwork[0].getNetworkWeightSize();
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

	m_cpuNetwork[0].setWeight(cpuNetworkWeight);
	m_gpuNetwork.setWeight(gpuNetworkWeight.data(), gpuNetworkWeight.size(), true);
	gpuNetworkWeight.clear();
	gpuNetworkWeight.resize(m_gpuNetwork.getNetworkWeightSize());
	m_gpuNetwork.getWeight(gpuNetworkWeight.data(), gpuNetworkWeight.size(), true);

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

	m_gpuNetwork.setWeight(testWeights.data(), testWeights.size(), true);
	m_gpuNetwork.runPredictions(testInput.data(), testInput.size(), true);

	std::vector<float> gpuResult(m_outputSize * m_modelNumber);
	std::vector<float> cpuResult(m_outputSize * m_modelNumber);
	m_gpuNetwork.getPredictions(gpuResult.data(), gpuResult.size(), true);

	std::vector<float> cpuWeight(m_cpuNetwork[0].getNetworkWeightSize());
	std::vector<float> cpuInput(m_inputSize);

	for (int i = 0; i < m_modelNumber; ++i)
	{
		std::copy(testWeights.begin() + i * cpuWeight.size(), testWeights.begin() + (i + 1) * cpuWeight.size(), cpuWeight.begin());
		std::copy(testInput.begin() + i * cpuInput.size(), testInput.begin() + (i + 1) * cpuInput.size(), cpuInput.begin());
		m_cpuNetwork[0].setWeight(cpuWeight);
		auto res = m_cpuNetwork[0].predict(cpuInput);
		for (int j = 0; j < res.size(); ++j)
		{
			cpuResult[i * m_outputSize + j] = res[j];
		}
	}

	int correctValues{};
	int incorrectValues{};
	for (int i = 0; i < gpuResult.size(); ++i)
	{
		//std::cout << "ID: " << i << "\tGPU: " << gpuResult[i] << "\tCPU : " << cpuResult[i] << std::endl;
		if (fabs(gpuResult[i] - cpuResult[i]) < 0.0001)
		{
			correctValues++;
		}
		else
		{
			//cout << "ID: " << i << "\tGPU: " << gpuResult[i] << "\tCPU : " << cpuResult[i] << std::endl;
			incorrectValues++;
		}
	}

	std::cout << "Prediction finished correct values: " << correctValues << " Incorrect values: " << incorrectValues << std::endl;
}


void compareArrays(std::string testName, const thrust::device_vector<float>& lhs, const thrust::device_vector<float>& rhs)
{
	std::vector<float> h_lhs(lhs.size());
	std::vector<float> h_rhs(rhs.size());

	cudaMemcpy(h_lhs.data(), thrust::raw_pointer_cast(lhs.data()), sizeof(float) * lhs.size(), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rhs.data(), thrust::raw_pointer_cast(rhs.data()), sizeof(float) * rhs.size(), cudaMemcpyDeviceToHost);

	int counter{};
	for (int i = 0; i < lhs.size(); ++i)
	{
		if (fabs(h_lhs[i] - h_rhs[i]) > 0.0001)
		{
			counter++;
		}
	}
	if (counter > 0)
	{
		std::cout << "Test: " << testName << " didint pass. Number of incorrect values: " << counter << std::endl;
	}
	else
	{
		std::cout << "Test: " << testName << " passed." << std::endl;
	}
}

void NetworkTester::testPredictionsSpeed()
{
	// Time
	Timer t;
	double gpuTime{};
	double gpuTimeFast{};
	double cpuTime{};
	// Set up data
	std::vector<float> testWeights(m_gpuNetwork.getNetworkWeightSize());
	std::vector<float> testInput(m_inputSize * m_modelNumber);
	thrust::device_vector<float> gpuTestInput(m_inputSize * m_modelNumber);
	std::vector<std::vector<float>> cpuWeight;
	std::vector<std::vector<float>> cpuInput;
	std::vector<float> gpuResultVec(m_outputSize * m_modelNumber);
	thrust::device_vector<float> gpuResultVecFast(m_outputSize * m_modelNumber);

	for (unsigned i = 0; i < m_testNumber; ++i)
	{
		std::cout << "Startin test: " << i << std::endl;
		// Results
		double cpuResult{};
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
			std::vector<float> modelWeight(testWeights.begin() + i * m_cpuNetwork[0].getNetworkWeightSize(), testWeights.begin() + (i + 1) * m_cpuNetwork[0].getNetworkWeightSize());
			m_cpuNetwork[i].setWeight(modelWeight);
			cpuInput.push_back(std::vector<float>(testInput.begin() + i * m_inputSize, testInput.begin() + (i + 1) * m_inputSize));
		}
		std::vector<std::vector<float>>results;
		t.start();
		for (int i = 0; i < m_modelNumber; ++i)
		{
			results.push_back(m_cpuNetwork[i].predict(cpuInput[i]));
		}
		t.stop();
		cpuTime += t.measure();
		for (int i = 0; i < m_modelNumber; ++i)
		{
			cpuResult += std::accumulate(results[i].begin(), results[i].end(), 0.0f);
		}
		m_gpuNetwork.setWeight(testWeights.data(), testWeights.size(), true);
		cudaDeviceSynchronize();
		{
			t.start();
			m_gpuNetwork.runPredictions(testInput.data(), testInput.size(), true);
			m_gpuNetwork.getPredictions(gpuResultVec.data(), gpuResultVec.size(), true);
			cudaDeviceSynchronize();
			t.stop();
			gpuTime += t.measure();
		}
		{
			t.start();
			m_gpuNetwork.runPredictions(thrust::raw_pointer_cast(gpuTestInput.data()), testInput.size(), false);
			m_gpuNetwork.getPredictions(thrust::raw_pointer_cast(gpuResultVecFast.data()), gpuResultVec.size(), false);
			cudaDeviceSynchronize();
			t.stop();
			gpuTimeFast += t.measure();
		}
		compareArrays("Test GPU with GPU FAST", gpuResultVec, gpuResultVecFast);
	}
	std::cout << "Time results" << std::endl;
	std::cout << "CPU TIME: " << cpuTime * 1000 / m_testNumber << std::endl;
	std::cout << "GPU TIME: " << gpuTime * 1000 / m_testNumber << std::endl;
	std::cout << "GPU TIME (WITHOUT TRANSFER)" << gpuTimeFast* 1000 / m_testNumber << std::endl;
}



void startNetworkTest()
{
	
	{
		size_t inputSize{ 24 };
		size_t outputSize{ 4 };
		size_t modelNumber{ 1000 };
		size_t testCount{ 10 };
		ActivationFunction outActFunc = ActivationFunction::SIGMOID;

		NetworkTester tester(inputSize, outputSize, modelNumber, testCount, outActFunc);

		tester.addLayer(18, ActivationFunction::RELU);
		tester.addLayer(18, ActivationFunction::RELU);
		tester.runTest();
	}

	{
		size_t inputSize{ 24 };
		size_t outputSize{ 4 };
		size_t modelNumber{ 10000 };
		size_t testCount{ 10 };
		ActivationFunction outActFunc = ActivationFunction::SIGMOID;

		NetworkTester tester(inputSize, outputSize, modelNumber, testCount, outActFunc);

		tester.addLayer(18, ActivationFunction::RELU);
		tester.addLayer(18, ActivationFunction::RELU);
		tester.runTest();
	}
	/*
	{
		size_t inputSize{ 784 };
		size_t outputSize{ 10 };
		size_t modelNumber{ 1000 };
		size_t testCount{ 10 };
		ActivationFunction outActFunc = ActivationFunction::SIGMOID;

		NetworkTester tester(inputSize, outputSize, modelNumber, testCount, outActFunc);

		tester.addLayer(5, ActivationFunction::RELU);
		tester.runTest();
	}
	{
		size_t inputSize{ 784 };
		size_t outputSize{ 10 };
		size_t modelNumber{ 10000 };
		size_t testCount{ 10 };
		ActivationFunction outActFunc = ActivationFunction::SIGMOID;

		NetworkTester tester(inputSize, outputSize, modelNumber, testCount, outActFunc);

		tester.addLayer(5, ActivationFunction::RELU);
		tester.runTest();
	}
	*/
	/*
	{
		size_t inputSize{ 100 };
		size_t outputSize{ 57 };
		size_t modelNumber{ 10'000 };
		size_t testCount{ 10 };
		ActivationFunction outActFunc = ActivationFunction::SIGMOID;

		NetworkTester tester(inputSize, outputSize, modelNumber, testCount, outActFunc);

		tester.addLayer(10, ActivationFunction::RELU);
		tester.addLayer(8, ActivationFunction::RELU);
		tester.runTest();
	}
	*/
	
	/*
	{
		size_t inputSize{ 20 };
		size_t outputSize{ 10 };
		size_t modelNumber{ 10'000 };
		size_t testCount{ 10 };
		ActivationFunction outActFunc = ActivationFunction::RELU;

		NetworkTester tester(inputSize, outputSize, modelNumber, testCount, outActFunc);
		tester.runTest();
	}
	*/
	
}