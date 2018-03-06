#include <iostream>
// #include <memory>
#include "util/logstream.h"

int main(int argc, char *argv[])
{
    std::shared_ptr<logstream> log = logstream::instance();

    *log << "Hello!" << std::endl;

    log->setType(los::debug1);
    *log << "Specialized Hello!" << std::endl;
    
    log->setType(los::debug2);
    *log << "Specialized Hello!" << std::endl;
}
