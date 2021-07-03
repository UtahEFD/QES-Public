#ifndef RANDOM_H
#define RANDOM_H

#include <iostream>
#include <cstdlib>
#include <math.h>

class random
{
public:
  random();
  static double uniRan();
  static double norRan();
  static double rangen();


private:
  static bool m_normal_value;
  static double m_remaining_value;
};
#endif
