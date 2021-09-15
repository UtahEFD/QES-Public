#include <iostream>
#include <cmath>
#include <vector>
#include <string>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "util/ParseException.h"
#include "util/ParseInterface.h"

#include "QESNetCDFOutput.h"

#include "handleWINDSArgs.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"
#include "WINDSOutputVisualization.h"
#include "WINDSOutputWorkspace.h"

#include "WINDSOutputWRF.h"

#include "TURBGeneralData.h"
#include "TURBOutput.h"

#include "Solver.h"
#include "CPUSolver.h"
#include "DynamicParallelism.h"
#include "GlobalMemory.h"
#include "SharedMemory.h"

#include "Sensor.h"

#include "TextTable.h"

int main(int argc, char *argv[])
{
  int gridSize[3] = { 100, 100, 100 };
  float gridRes[3] = { 1.0, 1.0, 1.0 };

  float a(0.0);
  // a = 2pi/Lz
  a = 2.0 * M_PI / (gridSize[2] * gridRes[2]);
  std::cout << a << std::endl;

  WINDSGeneralData *WGD = new WINDSGeneralData(gridSize, gridRes);
  TURBGeneralData *TGD = new TURBGeneralData(WGD);
  for (int k = 0; k < WGD->nz - 1; k++) {
    for (int j = 0; j < WGD->ny - 2; j++) {
      for (int i = 0; i < WGD->nx - 1; i++) {
        int faceID = i + j * WGD->nx + k * WGD->nx * WGD->ny;
        WGD->u[faceID] = sin(a * WGD->z[k]);
      }
    }
  }

  TGD->getDerivatives_v2(WGD);

  float error(0.0), numcell(0.0);

  for (std::vector<int>::iterator it = TGD->icellfluid.begin(); it != TGD->icellfluid.end(); ++it) {
    int cellID = *it;
    int k = (int)(cellID / ((WGD->nx - 1) * (WGD->ny - 1)));
    int j = (int)((cellID - k * (WGD->nx - 1) * (WGD->ny - 1)) / (WGD->nx - 1));
    int i = cellID - j * (WGD->nx - 1) - k * (WGD->nx - 1) * (WGD->ny - 1);
    //std::cout << WGD->z[k] << " " << TGD->Gxz[cellID]
    //          << " " << a * cos(a * WGD->z[k]) << std::endl;

    error += pow(TGD->Gxz[cellID] - a * cos(a * WGD->z[k]), 2.0);
    numcell++;
  }
  std::cout << "Euclidian distance = " << error << std::endl;
  std::cout << "RMSE = " << sqrt(error / numcell) << std::endl;


  return 0;
}
