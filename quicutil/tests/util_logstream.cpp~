#include <iostream>
#include <memory>
#include "util/logstream.h"

int main(int argc, char *argv[])
{
#ifdef WIN32
  logstream* log = logstream::instance();
#else
  std::tr1::shared_ptr<logstream> log = logstream::instance();
#endif

  *log << "Hello!" << std::endl;

  log->setType(los::debug1);
  *log << "Specialized Hello!" << std::endl;

  log->setType(los::debug2);
  *log << "Specialized Hello!" << std::endl;

}
