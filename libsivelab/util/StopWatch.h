#ifndef STOP_WATCH_H
#define STOP_WATCH_H 1

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include "Timer.h"

namespace sivelab
{
  class StopWatch
  {
    public:
      
      std::string name;
        
      StopWatch();
      StopWatch(std::string const& _name);
      
      virtual ~StopWatch();
      
      void start();
      void split();
      void stop();
      
      void reset();

      double getElapsedTime() const;
      
      Timer_t getStart() const;
      Timer_t getSplit(int) const;
      Timer_t getFinish() const;  
        
    protected:
      
      bool running;
      double totalRunTime;
      Timer* timer;
      std::vector<Timer_t> times;
  };
  std::ostream& operator<<(std::ostream&, StopWatch);
}


#endif
