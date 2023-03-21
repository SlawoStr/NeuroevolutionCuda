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
#include "CPU/MutationAlgo/NewMutation.h"
#include <src/Utility/Timer.h>


/*
int main()
{
    size_t modelNumber{ 20000 };
    size_t parentPairNumber{ 9900 };
    for (int i = 1; i <= 20; ++i)
    {
        NeuralNetwork network(30, 10, ActivationFunction::SIGMOID);
        network.addLayer(25, ActivationFunction::RELU);
        network.addLayer(15, ActivationFunction::RELU);
        Neuroevolution manager(network, modelNumber, parentPairNumber, i);
        manager.setSelection<WheelSelection>(modelNumber);
        manager.setCrossover<MPCCrossover>(0.9f);
        manager.setMutation<NewMutation>(1.0f, 0.1f);
        std::vector<std::pair<int, double>>  fitnessVec;
        for (int i = 0; i < modelNumber; ++i)
        {
            fitnessVec.push_back({ i,i * 2.0f });
        }
        Timer t;
        t.start();
        manager.run(fitnessVec);
        t.stop();
        std::cout << t.measure() * 1000 << std::endl;
    }
    return 0;
}
*/
