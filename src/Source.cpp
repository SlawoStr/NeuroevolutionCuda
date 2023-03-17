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
#include "CPU/MutationAlgo/AddMutation.h"
#include <src/Utility/Timer.h>

int main()
{
    /*
    Timer t;
    using namespace std;
    using namespace std::chrono;

    vector<int> values(10000);

    // Generate Random values
    auto f = []() -> int { return rand() % 10000; };

    // Fill up the vector
    generate(values.begin(), values.end(), f);

    // Get starting timepoint
    t.start();

    // Call the function, here sort()
    sort(values.begin(), values.end());

    // Get ending timepoint
    t.stop();

    // Get duration. Substart timepoints to
    // get duration. To cast it to proper unit
    // use duration cast method
    cout << "Time taken by function: "
        << t.measure() << " microseconds" << endl;

    return 0;
    */
    
    size_t modelNumber{ 5000 };
    size_t parentPairNumber{ 2400 };
    size_t threadNumber{ 1 };
    for (int i = 1; i <= 20; ++i)
    {
        NeuralNetwork network(30, 10, ActivationFunction::SIGMOID);
        network.addLayer(25, ActivationFunction::RELU);
        network.addLayer(15, ActivationFunction::RELU);

        Neuroevolution manager(network, modelNumber, parentPairNumber, i);
        manager.setSelection<TournamentSelection>(modelNumber, 5);
        manager.setCrossover<MPCCrossover>(0.9f);
        manager.setMutation<AddMutation>(0.9f, 0.1f, 0.25f, 0.50f);

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