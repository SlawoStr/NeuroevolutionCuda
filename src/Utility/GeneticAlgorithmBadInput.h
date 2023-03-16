#pragma once
#include <string>

class GeneticAlgorithmBadInput : public std::exception
{
public:
	GeneticAlgorithmBadInput(std::string message) : m_message(message) {}
	const char* what() const override
	{
		return m_message.c_str();
	}
private:
	std::string m_message;
};
