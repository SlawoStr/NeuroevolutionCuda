#pragma once
#include <vector>
#include <random>
#include "src/Utility/GeneticAlgorithmBadInput.h"

/// <summary>
/// Abstract class for selection algorithms
/// </summary>
class GeneticSelector
{
public:
	/// <summary>
	/// Prepare selection algorithm for next iteration
	/// </summary>
	/// <param name="modelFitness">Model fitness</param>
	/// <param name="randEngine">Random number</param>
	virtual void setSelector(const std::vector<std::pair<int, double>>& modelFitness) = 0;
	/// <summary>
	/// Generate parent pair
	/// </summary>
	/// <param name="randEngine">Initialized random engine</param>
	/// <returns>Parent pair</returns>
	virtual std::pair<int, int> getParent(const std::vector<std::pair<int, double>>& modelFitness, std::mt19937& randEngine) = 0;
};