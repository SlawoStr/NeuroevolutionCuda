#include "GPU/CudaNeuroevolution.cuh"
#include "GPU/SelectionAlgo/CudaTournamentSelection.cuh"
#include "GPU/SelectionAlgo/CudaTruncationSelection.cuh"
#include "GPU/SelectionAlgo/CudaWheelSelection.cuh"
#include "GPU/CrossoverAlgo/CudaSPCCrossover.cuh"
#include "GPU/CrossoverAlgo/CudaMPCCrossover.cuh"
#include "GPU/CrossoverAlgo/CudaUniformCrossover.cuh"
#include "GPU/MutationAlgo/CudaAddMutation.cuh"
#include "GPU/MutationAlgo/CudaSignMutation.cuh"
#include "GPU/MutationAlgo/CudaProcMutation.cuh"
#include "GPU/MutationAlgo/CudaNewMutation.cuh"
#include "src/Utility/Timer.h"
#include "src/Test/NetworkTester.cuh"


int main()
{
	// NETWORK TESTER
	//startNetworkTest();
	
	// Number of models
	size_t modelNumber{ 10000 };
	// Number of parents to generate in selection ( modelNumber - parentPairNumber * 2 = Elite selection)
	size_t parentPairNumber{ 4800 };
	// Create network
	CudaNeuralNetwork network(30, 10, modelNumber, ActivationFunction::SIGMOID);
	network.addLayer(25, ActivationFunction::RELU);
	network.addLayer(15, ActivationFunction::RELU);
	// Create neuroevolution
	CudaNeuroevolution manager(network, parentPairNumber, 480, 64);
	// Set Genetic operators
	manager.setSelection <CudaTournamentSelection>(modelNumber, 5);
	manager.setCrossover<CudaMPCCrossover>(0.9f);
	manager.setMutation<CudaProcMutation>(0.8f, 0.05f, 1.0f, 1.5f);

	std::vector<std::pair<int, double>>  fitnessVec;
	for (int i = 0; i < modelNumber; ++i)
	{
		fitnessVec.push_back({ i,i * 2.0f });
	}

	int testNumber{ 200 };
	Timer t;
	t.start();
	for (int i = 0; i < testNumber; ++i)
	{
		manager.run(fitnessVec);
	}
	cudaDeviceSynchronize();
	t.stop();
	std::cout << t.measure() * 1000 / testNumber << std::endl;
}

