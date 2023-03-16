#include "Layer.h"

/*                   Activation functions                              */
inline float TahnFunction(float x) { return tanhf(x); }
inline float SigmoidFunction(float x) { return 1.0f / (1.0f + expf(-x)); }
inline float ReluFunction(float x) { return x <= 0 ? 0 : x; }

////////////////////////////////////////////////////////////
Layer::Layer(size_t neuronNumber, size_t connectionNumber, ActivationFunction actFunc)
{
	m_neurons.resize(neuronNumber);
	// Add bias neuron
	if (connectionNumber > 0)
	{
		m_neurons.emplace_back();
		m_neurons.back().setValue(1.0f);
	}
	setActivationFunction(actFunc);
	setConnectionNumber(connectionNumber);
}

////////////////////////////////////////////////////////////
Layer::Layer(size_t neuronNumber, size_t connectionNumber) : m_actFunc{nullptr}
{
	m_neurons.resize(neuronNumber);
	// Add bias neuron
	if (connectionNumber > 0)
	{
		m_neurons.emplace_back();
		m_neurons.back().setValue(1.0f);
	}
	setConnectionNumber(connectionNumber);
}

////////////////////////////////////////////////////////////
void Layer::setConnectionNumber(size_t connectionNumber)
{
	m_connectionNumber = connectionNumber;
	for (auto& neuron : m_neurons)
	{
		neuron.setConnectionNumber(connectionNumber);
	}
}

////////////////////////////////////////////////////////////
void Layer::setActivationFunction(ActivationFunction actFunc)
{
	switch (actFunc)
	{
	case ActivationFunction::RELU:
		m_actFunc = ReluFunction;
		break;
	case ActivationFunction::TANH:
		m_actFunc = TahnFunction;
		break;
	case ActivationFunction::SIGMOID:
		m_actFunc = SigmoidFunction;
		break;
	}
}

////////////////////////////////////////////////////////////
void Layer::setNeuronValue(const std::vector<float>& input)
{
	for (int i = 0; i < input.size(); ++i)
	{
		m_neurons[i].setValue(input[i]);
	}
}

////////////////////////////////////////////////////////////
std::vector<float> Layer::getNeuronValue()
{
	std::vector<float> neuronValue;
	for (const auto& neuron : m_neurons)
	{
		neuronValue.push_back(neuron.getValue());
	}
	return neuronValue;
}

////////////////////////////////////////////////////////////
void Layer::feedForward(Layer* next)
{
	for (int i = 0; i < m_connectionNumber; ++i)
	{
		float sum{ 0.0f };
		for (const auto& neuron : m_neurons)
		{
			sum += neuron.getValue() * neuron[i];
		}
		next->m_neurons[i].setValue(next->m_actFunc(sum));
	}
}