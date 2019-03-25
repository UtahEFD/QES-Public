#include <iostream>

#include "util/handlePlumeArgs.h"
#include "quicloader/QUICProject.h"

using namespace sivelab;

int main(int argc, char *argv[])
{
  PlumeArgs quicArgs;
  quicArgs.process(argc, argv);

  QUICProject qproj( quicArgs.quicFile );

  for (int i=0; i<qproj.quBuildingData.buildings.size(); i++) {
      std::cout << "Building " << i+1 << " height: " << qproj.quBuildingData.buildings[i].height << std::endl;
  }
}

