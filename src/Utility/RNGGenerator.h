#pragma once
#include <random>

/// <summary>
/// Global random number generator
/// </summary>
class RNGGenerator
{
public:
	/// <summary>
	/// Generate reandom float in range [min,max)
	/// </summary>
	/// <typeparam name="T">Float precision</typeparam>
	/// <param name="min">Minimum value</param>
	/// <param name="max">Maximum value</param>
	/// <returns>Random float</returns>
	template<typename T>
	static T randFloat(T min, T max)
	{
		return std::uniform_real_distribution<T>(min, max)(m_rng);
	}

	/// <summary>
	/// Fill vector with random floats in range [min,max)
	/// </summary>
	/// <typeparam name="T">Vector precision</typeparam>
	/// <param name="min">Minimum value</param>
	/// <param name="max">Maximum value</param>
	/// <param name="arr">Vector of floats</param>
	template<typename T>
	static void randFloat(T min, T max, std::vector<T>& arr)
	{
		std::uniform_real_distribution<T>distr(min, max);
		for (T& val : arr)
		{
			val = distr(m_rng);
		}
	}

	/// <summary>
	/// Generate reandom int in range [min,max)
	/// </summary>
	/// <typeparam name="T">Int precision</typeparam>
	/// <param name="min">Minimum value</param>
	/// <param name="max">Maximum value</param>
	/// <returns>Random int</returns>
	template<typename T>
	static T randInt(T min, T max)
	{
		return std::uniform_int_distribution<T>(min, max - 1)(m_rng);
	}

	/// <summary>
	/// Generate reandom int in range [min,max)
	/// </summary>
	/// <typeparam name="T">Int precision</typeparam>
	/// <param name="min">Minimum value</param>
	/// <param name="max">Maximum value</param>
	/// <param name="arr">Vector of ints</param>
	template<typename T>
	static void randInt(T min, T max, std::vector<T>& arr)
	{
		std::uniform_int_distribution<T> distr(min, max - 1);
		for (T& val : arr)
		{
			val = distr(m_rng);
		}
	}

private:
	static std::mt19937 m_rng;       //!< Random number generator
};