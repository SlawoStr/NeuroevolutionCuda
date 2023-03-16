#include "CudaNeuralNetwork.cuh"
#include "src/Utility/NetworkInputExcepction.h"

CudaNeuralNetwork::CudaNeuralNetwork(size_t inputSize, size_t outputSize, size_t modelNumber, ActivationFunction outActFunc)
	: m_modelNumber{ modelNumber }
{
	if (inputSize == 0)
	{
		throw NetworkInputExcepction{ "Error: Network input layer size must be bigger than 0" };
	}
	if (outputSize == 0)
	{
		throw NetworkInputExcepction{ "Error: Network output layer size must be bigger than 0" };
	}
	if (modelNumber == 0)
	{
		throw NetworkInputExcepction{ "Error: Number of models must be bigger than 0" };
	}
	m_layers.push_back({ inputSize, outputSize, m_modelNumber });
	m_layers.push_back({ outputSize, 0, m_modelNumber,outActFunc });
}

void CudaNeuralNetwork::addLayer(size_t neuronNumber, ActivationFunction actFunc)
{
	if (neuronNumber == 0)
	{
		throw NetworkInputExcepction{ "Error: Number of neurons in layer must be bigger than 0" };
	}
	auto it = m_layers.insert(m_layers.end() - 1, CudaLayer(neuronNumber, m_layers.back().getNeuronsPerModel(), m_modelNumber, actFunc));
	(*--it).setConnections(neuronNumber);
}

void CudaNeuralNetwork::runPredictions(const float* ptr, size_t size, bool isHostAllocated, unsigned threadNumber)
{
	size_t layerSize = m_layers[0].getLayerSize();
	if (size != layerSize)
	{
		throw NetworkInputExcepction("Error: Array size must be equal to input layer size required: " + std::to_string(layerSize) + " received: " + std::to_string(size));
	}
	m_layers[0].setNeurons(ptr, isHostAllocated);
	for (int i = 0; i < m_layers.size() - 1; ++i)
	{
		m_layers[i].feedForward(&m_layers[i + 1], threadNumber);
	}
}

void CudaNeuralNetwork::getPredictions(float* ptr, size_t size, bool isHostAllocated)
{
	size_t layerSize = m_layers.back().getLayerSize();
	if (size != layerSize)
	{
		throw NetworkInputExcepction("Error: Array size must be equal to output layer size required: " + std::to_string(layerSize) + " received: " + std::to_string(size));
	}
	m_layers.back().getNeurons(ptr, isHostAllocated);
}

void CudaNeuralNetwork::getWeight(float* ptr, size_t size, bool isHostAllocated, unsigned threadNumber)
{
	size_t weightSize{};
	size_t modelWeight{};
	size_t layerOffset{};
	for (const auto& layer : m_layers)
	{
		weightSize += layer.getWeightSize();
	}
	modelWeight = weightSize / m_modelNumber;
	if (weightSize != size)
	{
		throw NetworkInputExcepction("Error: Array size must be equal to model weights required" + std::to_string(weightSize) + " reveived: " + std::to_string(size));
	}
	if (!isHostAllocated)
	{
		for (auto it = m_layers.begin(); it != m_layers.end() - 1; ++it)
		{
			(*it).getModelWeight(ptr + layerOffset, modelWeight, threadNumber);
			layerOffset += (*it).getWeightSize() / m_modelNumber;
		}
	}
	else
	{
		thrust::device_vector<float> devVec(size);
		cudaMemcpy(thrust::raw_pointer_cast(devVec.data()), ptr, sizeof(float) * size, cudaMemcpyHostToDevice);
		for (auto it = m_layers.begin(); it != m_layers.end() - 1; ++it)
		{
			(*it).getModelWeight(thrust::raw_pointer_cast(devVec.data()) + layerOffset, modelWeight, threadNumber);
			layerOffset += (*it).getWeightSize() / m_modelNumber;
		}
		cudaMemcpy(ptr, thrust::raw_pointer_cast(devVec.data()), sizeof(float) * size, cudaMemcpyDeviceToHost);
	}
}

void CudaNeuralNetwork::setWeight(const float* ptr, size_t size, bool isHostAllocated, unsigned threadNumber)
{
	size_t weightSize = getNetworkWeightSize();
	size_t modelWeight{};
	size_t layerOffset{};
	modelWeight = weightSize / m_modelNumber;
	if (weightSize != size)
	{
		throw NetworkInputExcepction("Error: Array size must be equal to model weights required" + std::to_string(weightSize) + " reveived: " + std::to_string(size));
	}
	if (!isHostAllocated)
	{
		for (auto it = m_layers.begin(); it != m_layers.end() - 1; ++it)
		{
			(*it).setModelWeight(ptr + layerOffset, modelWeight, threadNumber);
			layerOffset += (*it).getWeightSize() / m_modelNumber;
		}
	}
	else
	{
		thrust::device_vector<float> devVec(size);
		cudaMemcpy(thrust::raw_pointer_cast(devVec.data()), ptr, sizeof(float) * size, cudaMemcpyHostToDevice);
		for (auto it = m_layers.begin(); it != m_layers.end() - 1; ++it)
		{
			(*it).setModelWeight(thrust::raw_pointer_cast(devVec.data()) + layerOffset, modelWeight, threadNumber);
			layerOffset += (*it).getWeightSize() / m_modelNumber;
		}
	}
}

size_t CudaNeuralNetwork::getNetworkWeightSize()
{
	size_t weightSize{};
	for (const auto& layer : m_layers)
	{
		weightSize += layer.getWeightSize();
	}
	return weightSize;
}
