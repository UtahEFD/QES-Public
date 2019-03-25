#ifndef __QUIC_PROJECT_H__
#define __QUIC_PROJECT_H__ 1

#include <iostream>
#include <fstream>
#include <limits>
#include "util/fileHandling.h"

#include "quicloader/QUBuildings.h"
#include "quicloader/QUMetParams.h"
#include "quicloader/QUSimparams.h"

#include "quicloader/QPBuildout.h"
#include "quicloader/QPParams.h"
#include "quicloader/QPSource.h"
#include "quicloader/language_map.h"

namespace sivelab {

  // Class container to hold the various data for an entire QUIC
  // project.  The quic project path or proj file name is provided and
  // all subsequent files will be loaded.

  class QUICProject: public languageMap////languageMap 
  {
  public:
      QUICProject();          //directory
      QUICProject( const std::string& QUICProjectPath, bool beVerbose = true, bool readUrbOnly = false );
      ~QUICProject();

    //copy constructor
    QUICProject(const QUICProject& other)
      {
	std::cerr<<"Copy constructor called"<<std::endl;
	*this = other;

      }

    QUICProject& operator= (const QUICProject& other);

    std::string m_quicProjectPath;

    // Structures to contain the QU_simparams.inp, QU_buildings.inp, QU_metparams.inp files.
    quSimParams quSimParamData;
    quBuildings quBuildingData;
    quMetParams quMetParamData;
    
    // A structure to contain the QP_params.inp, QP_source.inp files.
    qpBuildout qpBuildoutData;
    qpParams qpParamData;
    qpSource qpSourceData;

    int majorVersion(void);
    int minorVersion(void);

    // Out of convenience, we store and copy these values from
    // quSimParamData
    int nx, ny, nz;
    float dx, dy, dz;
  
    bool readUrbOnly;
    bool beVerbose;

    // Kevin's TTM path variable
    std::string ttm_input_path;

  void initialize_quicProjecPath(std::string quicPath);        ///directory . same as constructor with string 
  /// void modify_value(std::string variable_name,std::string newvalue);    ////languageMap    
  // std::string retrieve(std::string variable_name); 
   void build_map(); ///languageMap            ///this is the function that builds the map for every data qu/qp datastructure
  private:
 
    bool isQUICProjFile(std::ifstream& inputStream);

    int m_majorVersionNumber, m_minorVersionNumber;
  };

}

#endif //  __QUIC_PROJECT_H__ 1
