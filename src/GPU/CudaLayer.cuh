#pragma once
#include <thrust/device_vector.h>
#include "src/Utility/NetworkEnum.h"

/// <summary>
/// Neural network layer containing neurons from all models
/// </summary>
class CudaLayer
{
public:
	/// <summary>
	/// Create layer with activation function
	/// </summary>
	/// <param name="neuronNumber">Number of neurons in layer</param>
	/// <param name="connectionNumber">Number of connection per each neuron( number of neurons in next layer except bias is exist)</param>
	/// <param name="modelNumber">Number of models</param>
	/// <param name="actFunc">Activation function</param>
	CudaLayer(size_t neuronNumber, size_t connectionNumber, size_t modelNumber, ActivationFunction actFunc);
	/// <summary>
	/// Create layer without activation function (input layer)
	/// </summary>
	/// <param name="neuronNumber">Number of neurons in layer</param>
	/// <param name="connectionNumber">Number of connection per each neuron( number of neurons in next layer except bias is exist)</param>
	/// <param name="modelNumber">Number of models</param>
	CudaLayer(size_t neuronNumber, size_t connectionNumber, size_t modelNumber);
	/// <summary>
	/// Set neurons in layer
	/// </summary>
	/// <param name="input">Input array containing values for all neurons in layer</param>
	/// <param name="isHostAllocated">Is array host allocated (allocated on cpu)</param>
	void setNeurons(const float* input, bool isHostAllocated);
	/// <summary>
	/// Get neurons values
	/// </summary>
	/// <param name="output">Output array with size that can fit all neurons values</param>
	/// <param name="isHostAllocated">Is array host allocated (allocated on cpu)</param>
	void getNeurons(float* output, bool isHostAllocated);
	/// <summary>
	/// Set number of connections in layer ( number of neurons in next layer )
	/// </summary>
	/// <param name="connectionNumber">Connections per neuron</param>
	void setConnections(size_t connectionNumber);
	/// <summary>
	/// Feed forward to next layer
	/// </summary>
	/// <param name="next">Pointer to next layer</param>
	/// <param name="threadNumber">Number of threads</param>
	void feedForward(CudaLayer* next);
	/// <summary>
	/// Get number of neurons per each model in this layer
	/// </summary>
	/// <returns>Number of neurons per model</returns>
	size_t getNeuronsPerModel()const { return m_neuronPerModel; }
	/// <summary>
	/// Get number of neurson in layer (from all models)
	/// </summary>
	/// <param name="bias">Is bias counted</param>
	/// <returns>Number of neurons in layer</returns>
	size_t getLayerSize() const;
	/// <summary>
	/// Get number of weights in layer from all models
	/// </summary>
	/// <returns>Number of weights</returns>
	size_t getWeightSize()const { return md_weights.size(); }
	/// <summary>
	/// Get model weight from layer and map it to global array
	/// </summary>
	/// <param name="output">Pointer to element in global array from which elements should be inserted</param>
	/// <param name="modelOffset">Number of weights of each model</param>
	void getModelWeight(float* output, size_t modelOffset);
	/// <summary>
	/// Set model weights from global array
	/// </summary>
	/// <param name="input">Pointer to element in global array from which elements should be copied</param>
	/// <param name="modelOffset">Number of weights of each model</param>
	void setModelWeight(const float* input, size_t modelOffset);
private:
	/// <summary>
	/// Initialize layer parameters
	/// </summary>
	void initLayer();
private:
	size_t m_neuronPerModel;					//!< Number of neurons per each model
	size_t m_connectionPerNeuron;				//!< Number of connection per each neuron
	size_t m_modelNumber;						//!< Number of models
	ActivationFunction m_actFunc;				//!< Activation function
	thrust::device_vector<float> md_neurons;	//!< Neuron device vector
	thrust::device_vector<float> md_weights;	//!< Weights device vector
	// Additional
	int m_threadNumber{ 32 };					//!< Number of threads for gpu kernel
	int m_blockNumber;							//!< Number of blocks for gpu kernel
	int m_batchSize{ 32 };						//!< Number of neuonr to process for each batch
	int m_weightPerBlock;						//!< Number of weights for each block
	int m_stride;								//!< Number of output neurons to process by last block
};