#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <chrono>


class TimerTool
{
public:
  // constructor
  TimerTool(std::string str_in) : timerName(str_in)
  {
    start = std::chrono::high_resolution_clock::now();
  }

  ~TimerTool()
  {}

  void stop()
  {
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = finish - start;
    std::cout << "Elapsed time for " << timerName << ": " << elapsed.count() << " s\n";
  }

private:
  TimerTool() {}

  std::string timerName;
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
};
