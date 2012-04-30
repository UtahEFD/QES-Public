#ifndef URB_STOPWATCH_LIST
#define URB_STOPWATCH_LIST 1

#include <vector>

#include "util/StopWatch.h"

class urbStopWatchList
{
  public:
    
    urbStopWatchList();
    virtual ~urbStopWatchList();
    
    double getTotalElapsedTime() const;
    double getSetupTime() const;
    double getIterationTime() const;
    
    void reset();
    void resetSetupTimers();
    void resetIterationTimers();
    
    // Setup timers
    StopWatch* parse;
		StopWatch* init;
		StopWatch* sort;
		StopWatch* sensor;
		StopWatch* bldngprm;
		StopWatch* intersect;
		StopWatch* bndrymat;
		  static const int stp_sz = 7;
		
		// Iteration timers
		StopWatch* malloc;
		StopWatch* trnsfr;
		StopWatch* diverg;
		StopWatch* denoms;
		StopWatch* comput;
		StopWatch* euler;
		StopWatch* diffus;
		  static const int ntl_sz = 7;

  private:
  
    std::vector<StopWatch*> stpList;
    std::vector<StopWatch*> ntlList;

};

#endif
