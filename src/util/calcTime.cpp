
#include "calcTime.h"

/***** public functions *****/

/*** constructor functions ***/
calcTime::calcTime()
{
}
/*** end constructor functions ***/

/*** reconstructor like functions ***/
void calcTime::reset()
{
  while (!storedTimerNames.empty()) {
    storedTimerNames.pop_back();
  }
}
/*** end reconstructor like functions ***/

/*** set value functions ***/
void calcTime::startNewTimer(const std::string &newTimerName)
{
  if (isTimerStored(newTimerName) == true) {
    std::cerr << "ERROR (calcTime::startNewTimer): Invalid newTimerName! newTimerName \"" << newTimerName
              << "\" already exists so can't start a new time with the same name! Exiting Program!" << std::endl;
    exit(1);
  }
  storedTimerNames.push_back(newTimerName);
  storedStartTimes.push_back(std::chrono::high_resolution_clock::now());// start recording execution time
}

void calcTime::resetStoredTimer(const std::string &timerName)
{
  // now find if timer exists and which index
  size_t foundTimerIdx = findTimerIdx(timerName);

  // restart the timer by setting the time at the index to a brand new timer value
  storedStartTimes.at(foundTimerIdx) = std::chrono::high_resolution_clock::now();
}
/*** end set value functions ***/

/*** get value functions ***/
void calcTime::printStoredTime(const std::string &timerName)
{
  // first set time this function is called to get a difference between the desired start time and the current time
  auto nowTime = std::chrono::high_resolution_clock::now();// finish recording execution time

  // now find if timer exists and which index
  size_t foundTimerIdx = findTimerIdx(timerName);

  // now get the stored time
  auto foundStartTime = storedStartTimes.at(foundTimerIdx);

  // now get the total number of seconds passed
  std::chrono::duration<double> elapsed = nowTime - foundStartTime;

  // now figure out how to convert total seconds into different time period amounts
  int passedMilliseconds;
  int passedSeconds = 0;
  int passedMinutes = 0;
  int passedHours = 0;
  int passedDays = 0;
  calcTimeDistribution(elapsed.count(), passedMilliseconds, passedSeconds, passedMinutes, passedHours, passedDays);

  // now print off whatever times are needed, depending on what is zero or not
  printTime(timerName, passedMilliseconds, passedSeconds, passedMinutes, passedHours, passedDays);
}
/*** end get value functions ***/


/***** private functions *****/


/*** utility functions ***/
bool calcTime::isTimerStored(const std::string &timerName)
{
  bool foundTimeName = false;
  for (size_t timerIdx = 0; timerIdx < storedTimerNames.size(); timerIdx++) {
    if (storedTimerNames.at(timerIdx) == timerName) {
      foundTimeName = true;
      break;
    }
  }

  return foundTimeName;
}

size_t calcTime::findTimerIdx(const std::string &timerName)
{
  bool foundTimerName = false;
  size_t foundTimerIdx = 0;
  for (size_t timerIdx = 0; timerIdx < storedTimerNames.size(); timerIdx++) {
    if (storedTimerNames.at(timerIdx) == timerName) {
      foundTimerName = true;
      foundTimerIdx = timerIdx;
      break;
    }
  }

  if (foundTimerName == false) {
    std::cerr << "ERROR (calcTime::findTimerIdx): Invalid timerName! timeName \""
              << timerName << "\" does not exist in calcTime storage! Exiting Program!" << std::endl;
    exit(1);
  }

  return foundTimerIdx;
}

void calcTime::calcTimeDistribution(const double &totalSeconds, int &milliseconds, int &seconds, int &minutes, int &hours, int &days)
{
  double secondsRemainder = totalSeconds;
  days = secondsRemainder / (24 * 3600);
  secondsRemainder = secondsRemainder - (24 * 3600 * days);
  hours = secondsRemainder / (3600);
  secondsRemainder = secondsRemainder - (3600 * hours);
  minutes = secondsRemainder / 60;
  secondsRemainder = secondsRemainder - (60 * minutes);
  seconds = secondsRemainder;
  milliseconds = (secondsRemainder - (seconds)) * 1000;
}

void calcTime::printTime(const std::string &timerName, const int &milliseconds, const int &seconds, const int &minutes, const int &hours, const int &days)
{
  std::cout << "  Elapsed time for \"" << timerName << "\" timer: ";
  if (days != 0) {
    std::cout << days << " days, ";
  }
  if (hours != 0) {
    std::cout << hours << " hours, ";
  }
  if (minutes != 0) {
    std::cout << minutes << " minutes, ";
  }
  std::cout << seconds << " seconds, " << milliseconds << " milliseconds" << std::endl;
}
/*** end utility functions ***/
