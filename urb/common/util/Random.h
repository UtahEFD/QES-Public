#ifndef __UBERPART_RANDOM_H__
#define __UBERPART_RANDOM_H__ 1

class Random 
{
 public:
  Random();
  Random(long seedval);

  // Returns a random number pulled from a uniform distribution.  The
  // value will be between 0 and 1.
  static float uniform();

  // Returns a random number pulled from a normal distribution with
  // mean 0 and standard deviation of 1.
  static float normal();

 private:
  static bool m_normal_value;
  static float m_remaining_value;
};

#endif // __UBERPART_RANDOM_H__

