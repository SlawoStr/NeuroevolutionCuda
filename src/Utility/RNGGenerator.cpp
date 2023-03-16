#include "RNGGenerator.h"

/// Initialize random device and generator
std::mt19937 RNGGenerator::m_rng(std::random_device{}());