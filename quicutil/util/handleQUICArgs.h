#ifndef __HANDLE_QUIC_ARGS_H__
#define __HANDLE_QUIC_ARGS_H__ 1

#include <iostream>
#include <string>

#include "ArgumentParsing.h"

namespace sivelab {

  class QUICArgs : public ArgumentParsing
  {
  public:
    QUICArgs();
    ~QUICArgs() {}

    void process(int argc, char *argv[]);

    bool verbose;
    std::string quicproj;
    std::string cycle;
    std::string surface;
    std::string ttm_input_path;     // KAB 09/05/14- Adding to point to TTM input deck path
    
    bool headless;
    bool fullscreen;
  };
  
}

#endif // __HANDLE_PLUME_ARGS_H__ 1
