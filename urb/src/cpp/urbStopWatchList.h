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
    sivelab::StopWatch* parse;
		sivelab::StopWatch* init;
		sivelab::StopWatch* sort;
		sivelab::StopWatch* sensor;
		sivelab::StopWatch* bldngprm;
		sivelab::StopWatch* intersect;
		sivelab::StopWatch* bndrymat;
		
		// Iteration timers
		sivelab::StopWatch* malloc;
		sivelab::StopWatch* trnsfr;
		sivelab::StopWatch* diverg;
		sivelab::StopWatch* denoms;
		sivelab::StopWatch* comput;
		sivelab::StopWatch* euler;
		sivelab::StopWatch* diffus;

  private:
  
    std::vector<sivelab::StopWatch*> stpList;
    std::vector<sivelab::StopWatch*> ntlList;

};

#endif
