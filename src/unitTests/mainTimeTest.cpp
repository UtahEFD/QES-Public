#include "util/QEStime.h"
#include <queue>

int main()
{
  QEStime time("2020-01-01T00:00");

  std::cout.precision(15);

  std::cout << "getTime() \t\t" << time.getTime() << std::endl;
  std::cout << "getEpochTime() \t" << time.getEpochTime() << std::endl;
  std::cout << "getTimestamp() \t" << time.getTimestamp() << std::endl;

  time.increment(10.0 * 3600.0 + 12.5);
  time.increment(.5);
  time.increment(.01);
  time.increment(.1);
  time.increment(.1);
  time.increment(.1);
  time.increment(.1);
  time.increment(.1);
  time.increment(.1);
  time.increment(.1);
  time.increment(20.0 * 3600.0 + 12.5);

  std::cout << "getTime() \t\t" << time.getTime() << std::endl;
  std::cout << "getEpochTime() \t" << time.getEpochTime() << std::endl;
  std::cout << "getTimestamp() \t" << time.getTimestamp() << std::endl;

  time += 10 * 60;

  std::cout << "getTime() \t\t" << time.getTime() << std::endl;
  std::cout << "getEpochTime() \t" << time.getEpochTime() << std::endl;
  std::cout << "getTimestamp() \t" << time.getTimestamp() << std::endl;

  time = time + 60 * 60 * 24;

  std::cout << "getTime() \t\t" << time.getTime() << std::endl;
  std::cout << "getEpochTime() \t" << time.getEpochTime() << std::endl;
  std::cout << "getTimestamp() \t" << time.getTimestamp() << std::endl;


  QEStime t1("2020-01-01T00:00");
  QEStime t2("2020-01-01T00:00");

  t1 += 60 * 60 * 24;

  std::cout << t1 - t2 << std::endl;
  std::cout << time - t2 << std::endl;

  QEStime t3;
  std::cout << t2 - t3 << std::endl;

  float dt = 0.1;

  QEStime ts("2020-01-01T00:00");
  QEStime tf = ts + 60 * 60;
  QEStime t = ts;

  std::cout << "ts \t" << ts << std::endl;
  std::cout << "tf \t" << tf << std::endl;

  std::cout << "--------------------------" << std::endl;
  QEStime nextPrompt = t + 5 * 60;
  while (t < tf) {

    t += dt;
    //if ((t % (5 * 60)) == 0.0) {
    //  std::cout << "t \t" << t.getTimestamp() << std::endl;
    //}

    if (t == nextPrompt) {
      std::cout << "t \t" << t << std::endl;
      nextPrompt = t + 5 * 60;
    }
  }
  std::cout << "--------------------------" << std::endl;

  std::cout << "t \t" << t.getTimestamp() << std::endl;
  std::cout << "tf \t" << tf << std::endl;
  std::cout << std::endl;


  ts = t1;
  std::queue<QEStime> sensor_times;
  for (int k = 0; k < 12; ++k) {
    sensor_times.push(ts + k * 5 * 60);
  }

  t = sensor_times.front();
  tf = sensor_times.back() + 5 * 60;

  std::cout << "total time \t" << tf - ts << std::endl;

  std::cout << "t \t" << t << std::endl;
  std::cout << "--------------------------" << std::endl;
  while (t < tf) {
    if (t == sensor_times.front()) {
      std::cout << "t \t" << t << std::endl;
      sensor_times.pop();
    }
    t += dt;
  }
  std::cout << "--------------------------" << std::endl;
  std::cout << "t \t" << t << std::endl;
  return 0;
}
