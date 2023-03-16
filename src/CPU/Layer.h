#pragma once
#include <vector>
#include <functional>
#include "Neuron.h"
#include "src/Utility/NetworkEnum.h"

/// <summary>
/// Neural network layer representation
/// </summary>
class Layer
{
public:
	/// <summary>
	/// Create layer
	/// </summary>
	/// <param name="neuronNumber">Number of neurons in layer</param>
	/// <param name="connectionNumber">Number of connection per each neuron ( number of neurons in next layer )</param>
	/// <param name="actFunc">Activation function</param>
	Layer(size_t neuronNumber, size_t connectionNumber, ActivationFunction actFunc);
	/// <summary>
	/// Create layer without activation function(input layer)
	/// </summary>
	/// <param name="neuronNumber">Number of neurons in layer</param>
	/// <param name="connectionNumber">Number of connection per each neuron ( number of neurons in next layer )</param>
	Layer(size_t neuronNumber, size_t connectionNumber);
	// Element Access
	Neuron& operator[](size_t index) { return m_neurons[index]; }
	const Neuron& operator[](size_t index) const { return m_neurons[index]; }
	Neuron& at(size_t index) { return m_neurons.at(index); }
	const Neuron& at(size_t index) const { return m_neurons.at(index); }
	// Iterators
	std::vector<Neuron>::iterator begin() { return m_neurons.begin(); }
	std::vector<Neuron>::iterator end() { return m_neurons.end(); }
	std::vector<Neuron>::const_iterator cbegin()const { return m_neurons.cbegin(); }
	std::vector<Neuron>::const_iterator cend()const { return m_neurons.end(); }
	/// <summary>
	/// Set number of connection for all neurons
	/// </summary>
	/// <param name="connectionNumber">Number of neuron connections (number of neurons in next layer)</param>
	void setConnectionNumber(size_t connectionNumber);
	/// <summary>
	/// Set activation function
	/// </summary>
	/// <param name="actFunc">Activation function enum</param>
	void setActivationFunction(ActivationFunction actFunc);
	/// <summary>
	/// Set neuron values
	/// </summary>
	/// <param name="input">Vector of inputs for layer neurons</param>
	void setNeuronValue(const std::vector<float>& input);
	/// <summary>
	/// Get values from all neurons in layer
	/// </summary>
	/// <returns>Vector of neuron values</returns>
	std::vector<float> getNeuronValue();
	/// <summary>
	/// Get number of neurons in layer
	/// </summary>
	/// <returns>Number of neurons</returns>
	size_t getNeuronNumber() const { return m_neurons.size(); }
	/// <summary>
	/// Get number of connections in each neuron
	/// </summary>
	/// <returns>Number of neuron connections ( one neuron )</returns>
	size_t getConnectionNumber()const { return m_connectionNumber; }
	/// <summary>
	/// Feed forward next layer
	/// </summary>
	/// <param name="next">Pointer to next layer</param>
	void feedForward(Layer* next);
private:
	std::function<float(float)>		m_actFunc;			//!< Pointer to activation function
	std::vector<Neuron>				m_neurons;			//!< Vector of neurons
	size_t							m_connectionNumber; //!< Number of connection per each neuron
};