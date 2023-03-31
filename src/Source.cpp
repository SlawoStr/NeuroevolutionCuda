#include "CPU/NeuralNetwork.h"
#include "GPU/CudaNeuralNetwork.cuh"
#include <iostream>
#include "CPU/SelectionAlgo/WheelSelection.h"
#include "CPU/Neuroevolution.h"
#include "CPU/SelectionAlgo/TournamentSelection.h"
#include "CPU/SelectionAlgo/TruncationSelection.h"
#include "CPU/SelectionAlgo/WheelSelection.h"
#include "CPU/CrossoverAlgo/SPCCrossover.h"
#include "CPU/CrossoverAlgo/MPCCrossover.h"
#include "CPU/CrossoverAlgo/UniformCrossover.h"
#include "CPU/MutationAlgo/AddMutation.h"
#include "CPU/MutationAlgo/ProcMutation.h"
#include "CPU/MutationAlgo/NewMutation.h"
#include "CPU/MutationAlgo/SignMutation.h"
#include <src/Utility/Timer.h>



int main()
{
    size_t modelNumber{ 5000 };
    size_t parentPairNumber{ 2400 };
    for (int i = 1; i <= 20; ++i)
    {
        // Create neural network
        NeuralNetwork network(30, 5, ActivationFunction::SIGMOID);
        network.addLayer(15, ActivationFunction::RELU);
        // Create neuroevolution
        Neuroevolution manager(network, modelNumber, parentPairNumber, i);
        // Set genetic operators
        manager.setSelection<TruncationSelection>(modelNumber, 0.3f);
        manager.setCrossover<SPCCrossover>(0.9f);
        manager.setMutation<AddMutation>(0.8f, 0.05f, -0.25f, 0.25f);
        std::vector<std::pair<int, double>>  fitnessVec;
        for (int i = 0; i < modelNumber; ++i)
        {
            fitnessVec.push_back({ i,i * 2.0f });
        }

        int testNumber{ 10 };
        Timer t;
        t.start();
        for (int i = 0; i < testNumber; ++i)
        {
            manager.run(fitnessVec);
        }
        t.stop();
        std::cout << t.measure() * 1000 / testNumber << std::endl;
    }
    return 0;
}
