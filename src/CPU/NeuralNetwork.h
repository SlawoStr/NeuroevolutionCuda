#pragma once
#include <vector>
#include "Layer.h"

/// <summary>
/// Neural network class
/// </summary>
class NeuralNetwork
{
public:
	/// <summary>
	/// Creat neural network with x input neurons and y output neurons with 'outActFunc' as output layer activation function
	/// </summary>
	/// <param name="inputSize">Number of input neurons</param>
	/// <param name="outputSize">Number of output neurons</param>
	/// <param name="outActFunc">Activation function enum</param>
	NeuralNetwork(size_t inputSize, size_t outputSize, ActivationFunction outActFunc);
	/// <summary>
	/// Add new layer (one before output layer)
	/// </summary>
	/// <param name="neuronNumber">Number of neurons in layer</param>
	/// <param name="actFunc">Layer activation function</param>
	void addLayer(size_t neuronNumber, ActivationFunction actFunc);
	/// <summary>
	/// Get number of layers in network
	/// </summary>
	/// <returns></returns>
	size_t getNetworkSize()const { return m_layers.size(); }
	/// <summary>
	/// Get number of all weights in network
	/// </summary>
	/// <returns></returns>
	size_t getNetworkWeightSize() const;
	/// <summary>
	/// Make predictions
	/// </summary>
	/// <param name="input">Vector of inputs for input layer</param>
	/// <returns></returns>
	std::vector<float> predict(const std::vector<float>& input);
	/// <summary>
	/// Get vector of weights from all layers
	/// </summary>
	/// <returns>Vector of weights</returns>
	std::vector<float> getWeight() const;
	/// <summary>
	/// Set weights of all layers
	/// </summary>
	/// <param name="weight">Weights vector</param>
	void setWeight(std::vector<float>& weight);
private:
	std::vector<Layer> m_layers;		//!<	Neural network layers
	size_t			   m_inputSize;		//!<	Number of neurons in input layer
	size_t			   m_outputSize;	//!<	Number of neuron in output layer
};