#pragma once
#include <random>

/// <summary>
/// Random number generator
/// </summary>
/// <typeparam name="T">Random number engine</typeparam>
template<typename T>
class RandomNumberGenerator
{
public:
	/// <summary>
	/// Create random number generator
	/// </summary>
	RandomNumberGenerator(int intMin,int intMax, float floatMin,float floatMax) : m_rng(std::random_device{}())
	{
		setIntDistribution(intMin, intMax);
		setFloatDistibution(floatMin, floatMax);
	}
	/// <summary>
	/// Set int distribution range [min,max)
	/// </summary>
	/// <param name="min">Minimum value</param>
	/// <param name="max">Maximum value</param>
	void setIntDistribution(int min, int max)
	{
		intDistr = std::uniform_int_distribution<int>(min, max - 1);
	}
	/// <summary>
	/// Set float distibution range [min,max)
	/// </summary>
	/// <param name="min">Minimum value</param>
	/// <param name="max">Maximum value</param>
	void setFloatDistibution(float min, float max)
	{
		floatDistr = std::uniform_real_distribution<float>(min, max);
	}
	/// <summary>
	/// Generate random int from set distribution
	/// </summary>
	/// <returns>Random int</returns>
	int randomInt() { return intDistr(m_rng); }
	/// <summary>
	/// Generate random int from set distribution
	/// </summary>
	/// <returns>Random float</returns>
	float randomFloat() { return floatDistr(m_rng); }

private:
	T m_rng;												//!< Random number generator
	std::uniform_int_distribution<int> intDistr;			//!< Int distribution
	std::uniform_real_distribution<float> floatDistr;		//!< Float distribution
};