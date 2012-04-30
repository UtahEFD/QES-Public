#include <iostream>
#include <math.h>
#ifndef WIN32
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#else
#include <windows.h>
#endif
#include "Random.h"

static float randVal()
{
#ifdef WIN32
  double rval = rand()/(double)RAND_MAX;  
  return rval;
#else
  return drand48(); 
#endif
}

Random::Random()
{
  // Seed the random number generator with a somewhat random (or at
  // least changing) value

#ifndef WIN32
  // On Unix systems, this should do it...
  srand48( time(0) % getpid() );
#else
  // Otherwise, just give a value until we figure out how to do it on
  // Windows.
  srand(2);
#endif

  m_normal_value = false;
}

Random::Random(long s)
{
  // Seed the random number generator with the value passed as an
  // argument. 

#ifndef WIN32
  // On Unix systems, this should do it...
  srand48( s );
#else
  // Otherwise, just give a value until we figure out how to do it on
  // Windows.
  srand( s );
#endif

  m_normal_value = false;
}

float Random::uniform()
{
  return randVal();
}

// The normal function returns a random number from a Gaussian
// distribution with mean 0 and standard deviation of 1.  This is
// accomplished by using the Box-Muller algorithm for transformation
// between different distributions.
float Random::normal()
{
  float rsq, v1, v2;

  if (m_normal_value == false)
    {
      do 
	{
	  v1 = 2.0f * randVal() - 1.0f;
	  v2 = 2.0f * randVal() - 1.0f;
	  rsq = v1*v1 + v2*v2;
	} while (rsq >= 1.0);
      
      rsq = sqrt( (-2.0f * log(rsq) ) / rsq );
      
      m_remaining_value = v2 * rsq;
      m_normal_value = true;

      return v1*rsq;
    }
  else
    {
      m_normal_value = false;
      return m_remaining_value;
    }
}

bool Random::m_normal_value;
float Random::m_remaining_value;
