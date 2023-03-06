#ifndef CALCTIME_H
#define CALCTIME_H

#include <iostream>
#include <vector>
#include <map>
#include <chrono>


class calcTime
{
public:
  // constructor
  calcTime();

  // reconstructor like functions
  void reset();

  // set value functions
  void startNewTimer(const std::string &newTimerName);
  void resetStoredTimer(const std::string &timerName);

  // get value functions
  void printStoredTime(const std::string &timerName);

private:
  // utility functions
  bool isTimerStored(const std::string &timerName);
  size_t findTimerIdx(const std::string &timerName);
  void calcTimeDistribution(const double &totalSeconds,
                            int &milliseconds,
                            int &seconds,
                            int &minutes,
                            int &hours,
                            int &days);
  void printTime(const std::string &timerName,
                 const int &milliseconds,
                 const int &seconds,
                 const int &minutes,
                 const int &hours,
                 const int &days);

  // data members
  std::vector<std::string> storedTimerNames;
  //std::vector<std::chrono::time_point<std::chrono::system_clock,std::chrono::duration<long,std::ratio<1,1000000000> > > > storedStartTimes;
  std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> storedStartTimes;

  std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> storedTimers;
};

#endif// CALCTIME_H
