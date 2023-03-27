#include <catch2/catch_test_macros.hpp>

#include "util/QEStime.h"
#include <queue>

TEST_CASE("Testing QEStime class")
{
  std::cout.precision(15);

  QEStime time("2020-01-01T00:00");
  // std::cout << "getEpochTime() \t" << time.getEpochTime() << std::endl;
  // std::cout << "getTimestamp() \t" << time.getTimestamp() << std::endl;

  REQUIRE(time.getTimestamp() == "2020-01-01T00:00:00");
  // REQUIRE(time.getEpochTime() == 1577836800);

  time.increment(10.0 * 3600.0 + 12.5);
  time.increment(20.0 * 3600.0 + 10 * 60 + 12.5);
  REQUIRE(time.getTimestamp() == "2020-01-02T06:10:25");

  time = time + 60 * 60 * 24;
  REQUIRE(time.getTimestamp() == "2020-01-03T06:10:25");

  QEStime t1("2020-01-01T00:00");
  t1 += 60 * 60 * 24;
  REQUIRE(t1.getTimestamp() == "2020-01-02T00:00:00");

  QEStime t2("2020-01-01T00:00");
  // std::cout << t1 - t2 << std::endl;
  REQUIRE(t1 - t2 == 60 * 60 * 24);

  t2 = t1;
  REQUIRE(t2.getTimestamp() == "2020-01-02T00:00:00");

  t2 = "2020-01-04T00:00:00";
  REQUIRE(t2.getTimestamp() == "2020-01-04T00:00:00");

  float dt = 0.1;
  QEStime ts("2020-01-01T00:00");
  QEStime tf = ts + 60 * 60;
  QEStime t = ts;

  // std::cout << "ts \t" << ts << std::endl;
  // std::cout << "tf \t" << tf << std::endl;

  // std::cout << "--------------------------" << std::endl;
  QEStime nextPrompt = t + 5 * 60;
  while (t < tf) {

    t += dt;

    if (t == nextPrompt) {
      // std::cout << "t \t" << t << std::endl;
      nextPrompt = t + 5 * 60;
    }
  }
  // std::cout << "--------------------------" << std::endl;

  // std::cout << "t \t" << t.getTimestamp() << std::endl;
  // std::cout << "tf \t" << tf << std::endl;
  REQUIRE(t.getTimestamp() == tf.getTimestamp());

  ts = "2020-01-01T00:00";
  std::queue<QEStime> sensor_times;
  for (int k = 0; k < 12; ++k) {
    sensor_times.push(ts + (float)k * 5 * 60);
  }

  // std::cout << "total time \t" << tf - ts << std::endl;

  dt = 10;

  int n = 0;
  t = sensor_times.front();
  // std::cout << "t \t" << t << std::endl;
  // std::cout << "--------------------------" << std::endl;
  do {
    QEStime timeNextWindUpdate;

    sensor_times.pop();
    if (sensor_times.empty())
      timeNextWindUpdate = t + 5 * 60;
    else
      timeNextWindUpdate = sensor_times.front();

    // td::cout << "t \t" << t << std::endl;
    // std::cout << "tf \t" << tf << std::endl;

    while (t < timeNextWindUpdate) {
      n++;
      t += dt;
    }

    // std::cout << "----" << std::endl;
  } while (!sensor_times.empty());
  // std::cout << "--------------------------" << std::endl;
  // std::cout << "t \t" << t << std::endl;
  // std::cout << "n \t" << n << std::endl;
  // std::cout << std::endl;
  REQUIRE(t.getTimestamp() == "2020-01-01T01:00:00");
  REQUIRE(n == 360);

  QEStime test;
  // std::cout << "t-now \t" << test << std::endl;
}
