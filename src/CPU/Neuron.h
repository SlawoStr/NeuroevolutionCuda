#pragma once
#include <vector>

/// <summary>
/// Neuron representation
/// </summary>
class Neuron
{
public:
	/// <summary>
	/// Create neuron connections
	/// </summary>
	/// <param name="connectionNumber">Number of connections (neurons in next layer)</param>
	void setConnectionNumber(size_t connectionNumber) { m_weights.resize(connectionNumber); }
	// Access to elements
	float& operator[](size_t index) { return m_weights[index]; }
	float operator[](size_t index)const { return m_weights[index]; }
	float& at(size_t index) { return m_weights.at(index); }
	float at(size_t index)const { return m_weights.at(index); }
	//Iterators
	std::vector<float>::iterator begin() { return m_weights.begin(); }
	std::vector<float>::iterator end() { return m_weights.end(); }
	std::vector<float>::const_iterator cbegin()const { return m_weights.cbegin(); }
	std::vector<float>::const_iterator cend()const { return m_weights.end(); }
	/// <summary>
	/// Get neuron value
	/// </summary>
	/// <returns>Neuron value</returns>
	float getValue() const { return m_value; }
	/// <summary>
	/// Set neuron value
	/// </summary>
	/// <param name="value">New neuron value</param>
	void setValue(float value) { m_value = value; }
private:
	std::vector<float>	m_weights;			//!< Neuron weights
	float				m_value{ 0.0f };	//!< Neuron value
};