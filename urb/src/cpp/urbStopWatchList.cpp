#include "urbStopWatchList.h"

urbStopWatchList::urbStopWatchList()
{
  // Setup Timers
  stpList.resize(stp_sz);
  stpList[0] = parse     = new StopWatch("File Parsing              ");			
	stpList[1] = init      = new StopWatch("Initialization            ");
	stpList[2] = sort      = new StopWatch("Building Sorting          ");
	stpList[3] = sensor    = new StopWatch("Sensors Initialization    ");
	stpList[4] = bldngprm  = new StopWatch("Building Parameterizations");
	stpList[5] = intersect = new StopWatch("Street Intersections      ");
	stpList[6] = bndrymat  = new StopWatch("Boundary Matrices         ");
	
	// Iteration timers
	ntlList.resize(ntl_sz);
	ntlList[0] = malloc = new StopWatch("Memory Allocation         ");
	ntlList[1] = trnsfr = new StopWatch("Memory Transfers          ");
	ntlList[2] = diverg = new StopWatch("Divergence Matrix Setup   ");
	ntlList[3] = denoms = new StopWatch("Denominators Setup        ");
	ntlList[4] = comput = new StopWatch("Iteration Time            ");
	ntlList[5] = euler  = new StopWatch("Euler Compute Time        ");
	ntlList[6] = diffus = new StopWatch("Diffusion Calculation Time");
}

urbStopWatchList::~urbStopWatchList()
{
  stpList.clear();
  ntlList.clear();

  // NOTE: FIX: Some problem here with corrupted memory.
  delete parse;
  delete init; 
  delete sort;
  delete sensor;
  delete bldngprm;
  delete intersect;
  delete bndrymat;
	delete malloc;
	delete trnsfr;
	delete diverg;
	delete denoms;
	delete comput;
	delete euler;
	delete diffus;
	
	// Kill danglers.
  parse = init = sort = sensor = bldngprm = intersect = bndrymat = 0;
	malloc = trnsfr = diverg = denoms = comput = euler = diffus = 0;
}
    
double urbStopWatchList::getTotalElapsedTime() const
{
  return this->getSetupTime() + this->getIterationTime();  
}

double urbStopWatchList::getSetupTime() const
{
  double ttl_tm = 0.0;
  
  for(int i = 0; i < stp_sz; i++)
  {
    ttl_tm += stpList[i]->getElapsedTime();
  }
  
  return ttl_tm;
}

double urbStopWatchList::getIterationTime() const
{
  double ttl_tm = 0.0;

  for(int i = 0; i < ntl_sz; i++)
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
  for(int i = 0; i < stp_sz; i++) {stpList[i]->reset();}
}

void urbStopWatchList::resetIterationTimers()
{
  for(int i = 0; i < ntl_sz; i++) {ntlList[i]->reset();}
}
