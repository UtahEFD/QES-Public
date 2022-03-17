#include "util/QEStime.h"


int main()
{
  QEStime time;
  time.setTimestamp("2020-01-01T00:00");

  std::cout << time.getTimestamp() << std::endl;

  time.increment(10.0 * 3600.0 + 12.5);
  time.increment(.5);
  time.increment(.1);
  time.increment(.1);
  time.increment(.1);
  time.increment(.1);
  time.increment(.1);
  time.increment(.1);
  time.increment(.1);
  time.increment(.1);
  time.increment(20.0 * 3600.0 + 12.5);

  std::cout << time.getTime() << std::endl;
  std::cout.precision(10);
  std::cout << time.getTime() << std::endl;

  std::cout << time.getEpochTime() << std::endl;
  std::cout << time.getTimestamp() << std::endl;

  QEStime t1;
  t1 = 10.5;
  std::cout << t1.getTime() << std::endl;

  QEStime t2;
  t2 = "2020-01-01T00:00";
  std::cout << t2.getTimestamp() << std::endl;

  t2 = time;
  std::cout << t2.getTimestamp() << std::endl;

  return 0;
}
