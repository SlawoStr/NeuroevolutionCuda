#pragma once
#include <chrono>

/// <summary>
/// Timer class to measure elapsed time
/// </summary>
class Timer
{
public:
    /// <summary>
    /// Save current time (start)
    /// </summary>
    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    /// <summary>
    /// Save current time (stop)
    /// </summary>
    void stop()
    {
        stop_time = std::chrono::high_resolution_clock::now();
    }

    /// <summary>
    /// Calculate elapsed time between start and stop
    /// </summary>
    /// <returns>Elapsed time</returns>
    double measure()
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() / 1000000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;              //!< Point in time when start was callled
    std::chrono::high_resolution_clock::time_point stop_time;               //!< Point in time when stop was called
}; 
