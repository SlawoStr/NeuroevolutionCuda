#include "GPU/CudaNeuroevolution.cuh"
#include "GPU/SelectionAlgo/CudaTournamentSelection.cuh"
#include "GPU/CrossoverAlgo/CudaSPCCrossover.cuh"
#include "GPU/CrossoverAlgo/CudaMPCCrossover.cuh"
#include "GPU/MutationAlgo/CudaAddMutation.cuh"
#include "GPU/MutationAlgo/CudaNewMutation.cuh"
#include "src/Utility/Timer.h"
#include "src/Test/NetworkTester.cuh"


int main()
{
	// NETWORK TESTER
	startNetworkTest();
	
	/*
	size_t modelNumber{ 20000 };
	size_t parentPairNumber{ 9900 };

	CudaNeuralNetwork network(30, 10, modelNumber, ActivationFunction::SIGMOID);
	network.addLayer(25, ActivationFunction::RELU);
	network.addLayer(15, ActivationFunction::RELU);

	CudaNeuroevolution manager(network, parentPairNumber, 480, 64);
	manager.setSelection < CudaTournamentSelection>(modelNumber, 5);
	manager.setCrossover<CudaMPCCrossover>(0.9f);
	//manager.setMutation<CudaAddMutation>(1.0f, 0.1f, -0.25f, 0.25f);
	manager.setMutation<CudaNewMutation>(0.8f, 0.05f);

	std::vector<std::pair<int, double>>  fitnessVec;
	for (int i = 0; i < modelNumber; ++i)
	{
		fitnessVec.push_back({ i,i * 2.0f });
	}

	for (int i = 0; i < 20; ++i)
	{
		Timer t;
		t.start();
		manager.run(fitnessVec);
		t.stop();
		std::cout << t.measure() * 1000 << std::endl;
	}
	*/
}