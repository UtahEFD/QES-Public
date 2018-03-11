#ifndef __HANDLE_PLUME_ARGS_H__
#define __HANDLE_PLUME_ARGS_H__ 1

#include <iostream>
#include <string>

#include "ArgumentParsing.h"

namespace sivelab {

class PlumeArgs : public ArgumentParsing
{
public:
  PlumeArgs();
  ~PlumeArgs() {}

  void process(int argc, char *argv[]);

  bool verbose;
  std::string quicFile;
  long numParticles;
  std::string concFile;
  int  concId;
  bool fullscreen;
  int networkMode;
  int viewingMode;
  char treadportView;
  // int dynamicTreadportFrustum;
  float sunAzimuth;
  float sunAltitude;
  bool onlyCalcShadows;
  bool offscreenRender;
  bool ignoreSignal;
  bool headless;
};

}

#endif // __HANDLE_PLUME_ARGS_H__ 1
