#include "StopWatch.h"

namespace sivelab
{
  StopWatch::StopWatch()
  {
    running = false;
    totalRunTime = 0.0;
    timer = new Timer(false);
    name = "N/A";
  }

  StopWatch::StopWatch(std::string const& _name)
  {
    running = false;
    totalRunTime = 0.0;
    timer = new Timer(false);
    name = _name;
  }

  StopWatch::~StopWatch()
  {
    delete timer;
  }

  void StopWatch::start()
  {
    if(!running)
    {
      times.clear();
      running = true;
      times.push_back(timer->tic());
    }
    else
    {
      // Already running message.
      std::cerr << "ERROR::Timer is already running." << std::endl;
    }
  }

  void StopWatch::split()
  {
    if(running) {times.push_back(timer->tic());}
  }

  void StopWatch::stop()
  {
    if(running)
    {
      running = false;
      times.push_back(timer->tic());
      
      // Only add to total time when a started timer has been stopped.
      // After that all the info should be cleared...right?
      totalRunTime += timer->deltas(this->getStart(), this->getFinish());
      //std::cout << "SW:" << name << ":" << totalRunTime << " secs." << std::endl;
    }
    else
    {
      // Already stopped message.
      std::cerr << "ERROR::Timer is already stopped." << std::endl;
    }
  }

  void StopWatch::reset()
  {
    running = false;
    times.clear();
    totalRunTime = 0.0;
  }

  double StopWatch::getElapsedTime() const
  {
    return totalRunTime;
  }

  Timer_t StopWatch::getStart() const
  {
    return (times.empty()) ? 0 : times[0] ;
  }

  Timer_t StopWatch::getSplit(int whch_splt) const
  {
    whch_splt %= times.size();
    
    return times[whch_splt];
  }

  Timer_t StopWatch::getFinish() const
  {
    return (times.empty()) ? 0 : times[times.size() - 1];
  }

  std::ostream& operator<<(std::ostream& out, StopWatch sw)
  {
    
    out.width(10); 
    out << sw.name << std::flush;
    out << " {" << std::flush;
    std::fixed(out); out.precision(6);
    out << sw.getElapsedTime() << " secs}" << std::flush;
    
    return out;
  }
}

