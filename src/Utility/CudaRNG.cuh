#pragma once
#include <curand.h>
#include <curand_kernel.h>

/// <summary>
/// Generate random int in range ['min','max')
/// </summary>
/// <param name="state">Curand state</param>
/// <param name="minValue">Minimum value</param>
/// <param name="maxValue">Maximum value</param>
/// <returns>Random int</returns>
__device__ int randomInt(curandState& state, int minValue, int maxValue);

/// <summary>
/// Generate random float in range['min',max')
/// </summary>
/// <param name="state">Curand state</param>
/// <param name="minValue">Minimum value</param>
/// <param name="maxValue">Maximum value</param>
/// <returns>Random float</returns>
__forceinline__ __device__ float randomFloat(curandState& state, float minValue, float maxValue)
{
	return curand_uniform(&state) * (maxValue - minValue) + minValue;
}

/// <summary>
/// Generate random states for threads
/// </summary>
/// <param name="blockNumbe">Number of blocks</param>
/// <param name="threadNumber">Number of threads</param>
/// <param name="seed">Random seed</param>
/// <returns>Initialized curand states</returns>
curandState* generateRandomStates(unsigned blockNumbe, unsigned threadNumber, unsigned seed);

/// <summary>
/// Generate random float in range['min',max') for array
/// </summary>
/// <param name="blockNumbe">Number of blocks</param>
/// <param name="threadNumber">Number of threads</param>
/// <param name="min">Minimum value</param>
/// <param name="max">Maximum value</param>
/// <param name="size">Size of array(number to generate)</param>
/// <param name="d_arr">Pointer to device array</param>
/// <param name="state">Random States for all threads</param>
void randomFloat(unsigned blockNumber, unsigned threadNumber, float min, float max, size_t size, float* d_arr, curandState* state);

/// <summary>
/// Generate random array of floats in range['min',max')
/// </summary>
/// <param name="blockNumbe">Number of blocks</param>
/// <param name="threadNumber">Number of threads</param>
/// <param name="min">Minimum value</param>
/// <param name="max">Maximum value</param>
/// <param name="size">Number of random numbers to generate</param>
/// <param name="state">random State</param>
/// <returns>Array of random floats</returns>
float* randomFloat(unsigned blockNumber, unsigned threadNumber, float min, float max, size_t size, curandState* state);

/// <summary>
/// Generate random int in range['min',max') for array
/// </summary>
/// <param name="blockNumbe">Number of blocks</param>
/// <param name="threadNumber">Number of threads</param>
/// <param name="min">Minimum value</param>
/// <param name="max">Maximum value</param>
/// <param name="size">Size of array(number to generate)</param>
/// <param name="d_arr">Pointer to device array</param>
/// <param name="state">Random States for all threads</param>
void randomInt(unsigned blockNumber, unsigned threadNumber, int min, int max, size_t size, int* d_arr, curandState* state);

/// <summary>
/// Generate random array of ints in range['min',max')
/// </summary>
/// <param name="blockNumbe">Number of blocks</param>
/// <param name="threadNumber">Number of threads</param>
/// <param name="min">Minimum value</param>
/// <param name="max">Maximum value</param>
/// <param name="size">Number of random numbers to generate</param>
/// <param name="state">random State</param>
/// <returns>Array of random ints</returns>
int* randomInt(unsigned blockNumber, unsigned threadNumber, int min, int max, size_t size, curandState* state);

