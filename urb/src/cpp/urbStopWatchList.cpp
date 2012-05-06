#include "urbStopWatchList.h"

urbStopWatchList::urbStopWatchList()
{
  // Setup Timers
  parse     = new sivelab::StopWatch("File Parsing              "); stpList.push_back(parse);
	init      = new sivelab::StopWatch("Initialization            "); stpList.push_back(init);
	sort      = new sivelab::StopWatch("Building Sorting          "); stpList.push_back(sort);
	sensor    = new sivelab::StopWatch("Sensors Initialization    "); stpList.push_back(sensor);
	bldngprm  = new sivelab::StopWatch("Building Parameterizations"); stpList.push_back(bldngprm);
	intersect = new sivelab::StopWatch("Street Intersections      "); stpList.push_back(intersect);
	bndrymat  = new sivelab::StopWatch("Boundary Matrices         "); stpList.push_back(bndrymat);
	
	// Iteration timers
	malloc = new sivelab::StopWatch("Memory Allocation         "); ntlList.push_back(malloc);
	trnsfr = new sivelab::StopWatch("Memory Transfers          "); ntlList.push_back(trnsfr);
	diverg = new sivelab::StopWatch("Divergence Matrix Setup   "); ntlList.push_back(diverg);
	denoms = new sivelab::StopWatch("Denominators Setup        "); ntlList.push_back(denoms);
	comput = new sivelab::StopWatch("Iteration Time            "); ntlList.push_back(comput);
	euler  = new sivelab::StopWatch("Euler Compute Time        "); ntlList.push_back(euler);
	diffus = new sivelab::StopWatch("Diffusion Calculation Time"); ntlList.push_back(diffus);
}

urbStopWatchList::~urbStopWatchList()
{
  // TODO Fix this.
  // NOTE: FIX: Some problem here with corrupted memory.
  /*
  for (unsigned i = 0; i < stpList.size(); i++)
  {
    delete stpList[i];
    stpList[i] = 0;
  }
  stpList.clear();
  
  for (unsigned i = 0; i < ntlList.size(); i++)
  {
    delete ntlList[i];
    ntlList[i] = 0;
  }
  ntlList.clear();
  */
}
    
double urbStopWatchList::getTotalElapsedTime() const
{
  return this->getSetupTime() + this->getIterationTime();  
}

double urbStopWatchList::getSetupTime() const
{
  double ttl_tm = 0.0;
  
  for(unsigned i = 0; i < stpList.size(); i++)
  {
    ttl_tm += stpList[i]->getElapsedTime();
  }
  
  return ttl_tm;
}

double urbStopWatchList::getIterationTime() const
{
  double ttl_tm = 0.0;

  for(unsigned i = 0; i < ntlList.size(); i++)
  {
    ttl_tm += ntlList[i]->getElapsedTime();
  }
  
  return ttl_tm;
}

void urbStopWatchList::reset()
{
  this->resetSetupTimers();
  this->resetIterationTimers();
}

void urbStopWatchList::resetSetupTimers()
{
  for(unsigned i = 0; i < stpList.size(); i++) {stpList[i]->reset();}
}

void urbStopWatchList::resetIterationTimers()
{
  for(unsigned i = 0; i < ntlList.size(); i++) {ntlList[i]->reset();}
}
