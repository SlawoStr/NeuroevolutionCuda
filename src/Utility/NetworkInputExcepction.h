#pragma once
#include <string>

class NetworkInputExcepction : public std::exception
{
public:
	NetworkInputExcepction(std::string message) : m_message(message) {}
	const char* what() const override
	{
		return m_message.c_str();
	}
private:
	std::string m_message;
};
