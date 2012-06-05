#include <iostream>
#include <cstdlib>
#include <string>

#include "util/handleQUICArgs.h"
#include "quicutil/QUICProject.h"

struct WindFieldDomainData
{
  float x, y, z;
  float u, v, w;
};
void loadQUICWindField(int nx, int ny, int nz, const std::string &quicFilesPath, std::vector<WindFieldDomainData>& windFieldData);

sivelab::QUICProject *data;

int main(int argc, char** argv) 
{
  // Argument parsing is handled via the utility libraries in the
  // libsivelab codebase.
  sivelab::QUICArgs quicArgs;
  quicArgs.process( argc, argv );

  // Create a new QUIC data object using the libsivelab codebase for
  // loading QUIC data.
  // 
  // To run the code and provide the quicproj file, you can do this:
  //
  // ./plume -q <REPLACE WITH YOUR PATH>/quicdata/SBUE_small_bldg/SBUE_small_bldg.proj 

  data = new sivelab::QUICProject( quicArgs.quicproj );
  std::cout << "Done loading QUIC data.\n" << std::endl;

  // You could work to load the data in QUICProject into CUDA memory...
  // 1. Wind velocities - domain data
  //       While you wait to load actual wind data... why not just create random wind data???  drand48()
  // 2. Find number of particles in qpParamData structure of QUICProject, and then create CUDA memory to 
  //    hold the particles...

  // For the time being, the following code will load the Wind
  // Velocities and the Turbulence fields of the domain.  I (Pete) am
  // working to abstract this and nicely place it in quicutil.
  std::vector<WindFieldDomainData> windFieldData( data->nx * data->ny * data->nz );
  loadQUICWindField(data->nx, data->ny, data->nz, data->m_quicProjectPath, windFieldData);

  //
  // 3.  Build kernel to simply advect the particles...
  //
  // 4. Use time step in QPParams.h to determine the
  // loop... while (simulation duration is not yet complete) run
  // advection kernel again...

  delete data;
}


void loadQUICWindField(int nx, int ny, int nz, const std::string &quicFilesPath, std::vector<WindFieldDomainData>& windFieldData)
{
  // 
  // for now, this only loads the ascii files... binary will be
  // incorporated into quicutil
  //

  assert( quicFilesPath.c_str() != NULL );
  std::string path = quicFilesPath + "QU_velocity.dat";

  std::ifstream QUICWindField;
  QUICWindField.open(path.c_str()); //opening the wind file  to read

  if(!QUICWindField){
    std::cerr<<"Unable to open QUIC Windfield file : QU_velocity.dat ";
    exit(1);
  }

  std::string header;  // I am just using a very crude method to read the header of the wind file
  QUICWindField>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header>>header;
  QUICWindField>>header>>header>>header>>header>>header;

  // Note that the first layer of the wind velocities are the
  // sub-ground layer (or the layer just below the ground!  In other
  // words, k=0 is the layer below the ground; k=1 is the layer above
  // the ground.

  // there are 6 columns in the wind file (posx, posy, posz, u, v, w)

  double quicIndex;

  for(int k = 0; k < nz; k++){   
    for(int i = 0; i < ny; i++){
      for(int j = 0; j < nx; j++){
	int p2idx = k*nx*ny + i*nx + j;

	QUICWindField >> windFieldData[p2idx].x;
	QUICWindField >> windFieldData[p2idx].y;
	QUICWindField >> windFieldData[p2idx].z;
	QUICWindField >> windFieldData[p2idx].u;
	QUICWindField >> windFieldData[p2idx].v;
	QUICWindField >> windFieldData[p2idx].w;

	std::cout << "WindField [" << p2idx << "] = (" 
		  << windFieldData[p2idx].x << ' '
		  << windFieldData[p2idx].y << ' '
		  << windFieldData[p2idx].z << ' '
		  << windFieldData[p2idx].u << ' '
		  << windFieldData[p2idx].v << ' '
		  << windFieldData[p2idx].z << ")" << std::endl;
      }
    }
  }
  
  QUICWindField.close();
}
