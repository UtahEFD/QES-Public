#include <iostream>

#include "util/handlePlumeArgs.h"
#include "quicutil/QUICProject.h"

#include "quicutil/QPTurbulenceField.h"

using namespace sivelab;

int main(int argc, char *argv[])
{
  PlumeArgs quicArgs;
  quicArgs.process(argc, argv);

  QUICProject qproj( quicArgs.quicFile );

  quSimParams quSim;
  quSim.readQUICFile("QU_simparams.inp");
  std::cout << "Domain: " << quSim.nx << " X " << quSim.ny << " X " << quSim.nz << std::endl;

  qpTurbulenceField qpTurbAscii( quSim.nx, quSim.ny, quSim.nz );
  qpTurbAscii.readQUICFile("QP_turbfield.dat");

  qpTurbulenceField qpTurbBin( quSim.nx, quSim.ny, quSim.nz );
  qpTurbBin.readQUICFile("QP_turbfield.bin");
}
