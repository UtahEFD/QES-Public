#ifndef __QUIC_PROJECT_H__
#define __QUIC_PROJECT_H__ 1

#include <iostream>
#include <fstream>
#include <limits>
#include "util/fileHandling.h"

#include "quicutil/QUBuildings.h"
#include "quicutil/QUMetParams.h"
#include "quicutil/QUSimparams.h"

#include "quicutil/QPBuildout.h"
#include "quicutil/QPParams.h"
#include "quicutil/QPSource.h"

namespace sivelab {

  // Class container to hold the various data for an entire QUIC
  // project.  The quic project path or proj file name is provided and
  // all subsequent files will be loaded.

  class QUICProject
  {
  public:
    QUICProject( const std::string& QUICProjectPath, bool beVerbose = true, bool readUrbOnly = false );
    ~QUICProject() {}

    std::string m_quicProjectPath;

    // Structures to contain the QU_simparams.inp, QU_buildings.inp, QU_metparams.inp files.
    quSimParams quSimParamData;
    quBuildings quBuildingData;
    quMetParams quMetParamData;
    
    // A structure to contain the QP_params.inp, QP_source.inp files.
    qpBuildout qpBuildoutData;
    qpParams qpParamData;
    qpSource qpSourceData;

    // Out of convenience, we store and copy these values from
    // quSimParamData
    int nx, ny, nz;
    float dx, dy, dz;
  
    bool readUrbOnly;
    bool beVerbose;
  
  private:
    QUICProject() {}
    bool isQUICProjFile(std::ifstream& inputStream);

  };

}

#endif //  __QUIC_PROJECT_H__ 1
