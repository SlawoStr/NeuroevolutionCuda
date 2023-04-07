# NeuroevolutionCuda

Neuroevolution is a type of machine learning that involves evolving neural networks using genetic algorithms. In this approach, a population of neural networks is initialized and evaluated on a given task. The fittest networks are then selected for reproduction, and genetic operators such as crossover and mutation are applied to create new networks. This process is repeated over multiple generations until the population converges to an optimal solution. 

# Implemented algorithms

Selection Algorithms:

- Truncation Selection ( Insufficient computation to fully utilize gpu power ) 
- Wheel Selection (Warp divergence problem ( CPU execution is much better ))
- Tournament Selection

Crossover Algorithms:

- Single point crossover
- Two point crossover
- Uniform crossover

Mutation Algorithms:

- Add random value to weight
- Generate new weight value
- Change weight sign
- Scale weight by a value

# Usage

```c
// Number of models
size_t modelNumber{ 20000 };
// Number of parents to generate in selection ( modelNumber - parentPairNumber * 2 = Elite selection)
size_t parentPairNumber{ 9900 };
// Create network
CudaNeuralNetwork network(30, 10, modelNumber, ActivationFunction::SIGMOID);
network.addLayer(25, ActivationFunction::RELU);
network.addLayer(15, ActivationFunction::RELU);
// Create neuroevolution
CudaNeuroevolution manager(network, parentPairNumber, 480, 64);
// Set Genetic operators
manager.setSelection < CudaTournamentSelection>(modelNumber, 5);
manager.setCrossover<CudaMPCCrossover>(0.9f);
manager.setMutation<CudaNewMutation>(0.8f, 0.05f);
// Run neuroevolution
manager.run(fitnessVec);
```

# Performance v1.0

Configuration:   

||Config 1|Config 2|Config 3|Config 4|  
|---|---|---|---|---|
|*Population Size*|5000|5000|5000|5000|
|Parent Number|4800|4800|4800|4800|
|Network topology|30-15-5|30-15-5|30-25-15-10|30-25-15-10|
|Selection|Truncation|Tournament|Roulette|Tournament|
|Crossover|SPC|MPC|Uniform|MPC|
|Mutation|ADD|NEW|SIGN|PROC|

Results: 

||Config 1|Config 2|Config 3|Config 4|  
|---|---|---|---|---|
|*CPU Multithread*|4.49ms|3.92ms|14.40ms|11.06ms|
|GPU|0.56ms|0.45ms|3.90ms|0.95ms|






