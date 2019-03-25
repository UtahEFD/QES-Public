#include "QUVelocities.h"

#include <iomanip>

#include "velocities.h"
#include "QUSimparams.h"

bool quVelocities::readQUICFile(const std::string &filename)
{
  std::cerr << "Error in " << __FILE__ << ":" << __func__ << std::endl;
  std::cerr << "QUVelocities does not have implementation to read from file." << std::endl;
  return false;
}

bool quVelocities::writeQUICFile(const std::string &filename)
{
  std::cerr << "Error in " << __FILE__ << ":" << __func__ << std::endl;
  std::cerr << "QUVelocities requires more paramaters.  Use writeQUICFile with filename, velocities, and simParams" << std::endl;
  return false;
}
	
bool quVelocities::writeQUICFile(const std::string &filename, const QUIC::velocities &vlcts, const quSimParams &simParams)
{
  std::cout << "Writing QU_velocities.dat..." << std::flush;

  using namespace std;

  ofstream writeWind(filename.c_str());

  if(!writeWind.is_open())	
  {
    cerr << "failed to open " + filename + "." << flush;
    return false;
  }

  // Write first five lines
  writeWind << "%matlab like velocity output file from urbModule" << endl;
  writeWind << "\t" << simParams.num_time_steps << "\t!Number of time steps" << endl;

  int grid_row = vlcts.dim.x;
  int grid_slc = vlcts.dim.x*vlcts.dim.y;

  int nx = vlcts.dim.x - 1;
  int ny = vlcts.dim.y - 1;
  int nz = vlcts.dim.z - 1;

  for(int ts = 0; ts < simParams.num_time_steps; ts++)
  {
    writeWind << " %Begin Output for new time step" << endl;
    writeWind << "\t" << simParams.time_incr << "!Time Increment" << endl;
    
    writeWind << fixed;
    writeWind.precision(5);
    writeWind << setw(12) << (simParams.start_time + ts*simParams.time_incr) << flush;
    writeWind << setw(12) << "!Time" << endl;
    
    for(int k = 0; k < nz; k++)
    for(int j = 0; j < ny; j++)
    for(int i = 0; i < nx; i++)
    {
      int ndx    = k*grid_slc + j*grid_row + i;
      int ndx_pi = ndx + 1;
      int ndx_pj = ndx + grid_row;
      int ndx_pk = ndx + grid_slc;
      
      // Write out indices.
      writeWind << setw(12) << (i + .5);//*dx;
      writeWind << setw(12) << (j + .5);//*dy;
      writeWind << setw(12) << (k - .5);//*dz;
      
      // Write out wind values.
      writeWind << setw(12) << .5*(vlcts.u[ndx] + vlcts.u[ndx_pi]);
      writeWind << setw(12) << .5*(vlcts.v[ndx] + vlcts.v[ndx_pj]);
      writeWind << setw(12) << .5*(vlcts.w[ndx] + vlcts.w[ndx_pk]);
      writeWind << endl;
    }
  }
  writeWind.close();
  
  std::cout <<"done.\n" << std::flush;

  return true; // Check for file errors...
}
