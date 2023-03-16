#include "NeuralNetwork.h"
#include "src/Utility/NetworkInputExcepction.h"

////////////////////////////////////////////////////////////
NeuralNetwork::NeuralNetwork(size_t inputSize, size_t outputSize, ActivationFunction outActFunc) : m_inputSize{ inputSize }, m_outputSize{ outputSize }
{
	if (inputSize == 0)
	{
		throw NetworkInputExcepction{ "Error: Network input layer size must be bigger than 0" };
	}
	if (outputSize == 0)
	{
		throw NetworkInputExcepction{ "Error: Network output layer size must be bigger than 0" };
	}
	m_layers.emplace_back(m_inputSize, m_outputSize);
	m_layers.emplace_back(m_outputSize, 0, outActFunc);
}

////////////////////////////////////////////////////////////
void NeuralNetwork::addLayer(size_t neuronNumber, ActivationFunction actFunc)
{
	if (neuronNumber == 0)
	{
		throw NetworkInputExcepction("Error: Size of network hidden layer muse be bigger than 0");
	}
	auto it = m_layers.insert(m_layers.end() - 1, Layer{ neuronNumber,m_outputSize,actFunc });
	std::prev(it)->setConnectionNumber(neuronNumber);
}

////////////////////////////////////////////////////////////
size_t NeuralNetwork::getNetworkWeightSize() const
{
	size_t weightSize{};
	for (const auto& layer : m_layers)
	{
		weightSize += layer.getNeuronNumber() * layer.getConnectionNumber();
	}
	return weightSize;
}

////////////////////////////////////////////////////////////
std::vector<float> NeuralNetwork::predict(const std::vector<float>& input)
{
	if (input.size() != m_inputSize)
	{
		throw NetworkInputExcepction("Error: Required input size: " + std::to_string(m_inputSize) + "\t received input size: " + std::to_string(input.size()));
	}
	// Start prediction
	m_layers[0].setNeuronValue(input);
	for (auto it = m_layers.begin(); it != m_layers.end() - 1; ++it)
	{
		it->feedForward(&(*std::next(it)));
	}
	// Return output layer neuron values
	return m_layers.back().getNeuronValue();
}

////////////////////////////////////////////////////////////
std::vector<float> NeuralNetwork::getWeight() const
{
	std::vector<float> weight;

	for (const auto& layer : m_layers)
	{
		for (int i = 0; i < layer.getConnectionNumber(); ++i)
		{
			for (auto it = layer.cbegin(); it != layer.cend(); ++it)
			{
				weight.push_back((*it)[i]);
			}
		}
	}
	return weight;
}

////////////////////////////////////////////////////////////
void NeuralNetwork::setWeight(std::vector<float>& weight)
{
	if (weight.size() != getNetworkWeightSize())
	{
		throw NetworkInputExcepction("Error: Vector of weights have " + std::to_string(weight.size()) + " weights Required: " + std::to_string(getNetworkWeightSize()));
	}
	auto weightIt = weight.begin();
	for (auto& layer : m_layers)
	{
		for (int i = 0; i < layer.getConnectionNumber(); ++i)
		{
			for (auto it = layer.begin(); it != layer.end(); ++it)
			{
				(*it)[i] = *weightIt++;
			}
		}
	}
}
