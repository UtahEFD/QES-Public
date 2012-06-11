/*
 *  test_Timer.cpp
 *
 *  Created by Pete Willemsen on 10/6/09.
 *  Copyright 2009 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 * This file is part of SIVE Lab library (libsive).
 *
 */

#include <iostream>
#include <cmath>
#include <vector>
#include "util/Timer.h"

using namespace sivelab;

int main(int argc, char *argv[])
{
#ifdef WIN32
	std::cout << "Not yet supported on Windows." << std::endl;
#else
  std::vector<long> sleepTimes;

  srand48( time(0) % getpid() );

  for (int i=1; i<=10; i++)
    sleepTimes.push_back( i*10000 );

  for (int i=0; i<10; i++)
    sleepTimes.push_back( (int)floor(drand48() * 10000) );

  long avgDiff = 0;

  Timer t0;
  for (unsigned int i=0; i<sleepTimes.size(); i++)
    {
      Timer_t startTime = t0.tic();
      usleep( sleepTimes[i] );
      Timer_t endTime = t0.tic();
      Timer_t diffTime = t0.deltau(startTime, endTime);
      std::cout << "Low-res timer result after usleeping for " << sleepTimes[i] << " microseconds --> result = " << diffTime << " us [diff = " << diffTime - sleepTimes[i] << "]" << std::endl;
      avgDiff += (diffTime - sleepTimes[i]);
    }
  avgDiff /= sleepTimes.size();
  std::cout << "Average difference between specified usleep and measured time: " << avgDiff << " us (" << avgDiff/1000000.0 << " s)" << std::endl << std::endl;

  avgDiff = 0;
  Timer t1(true);
  for (unsigned int i=0; i<sleepTimes.size(); i++)
    {
      Timer_t startTime = t1.tic();
      usleep( sleepTimes[i] );
      Timer_t endTime = t1.tic();
      Timer_t diffTime = t1.deltau(startTime, endTime);
      std::cout << "High-res timer result after usleeping for " << sleepTimes[i] << " microseconds --> result = " << diffTime << " us [diff = " << diffTime - sleepTimes[i] << "]" << std::endl;
      avgDiff += (diffTime - sleepTimes[i]);
    }
  avgDiff /= sleepTimes.size();
  std::cout << "Average difference between specified usleep and measured time: " << avgDiff << " us (" << avgDiff/1000000.0 << " s)" << std::endl << std::endl;
#endif
}
