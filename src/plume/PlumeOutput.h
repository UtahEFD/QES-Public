//
//  NetCDFOutput.h
//
//  This class handles saving output files for Eulerian binned Lagrangian particle data,
//   where this class handles the binning of the Lagrangian particle data
//  This is a specialized output class derived
//   and inheriting from QESNetCDFOutput.
//
//  Created by Fabien Margairaz on 01/25/20
//  Modified by Loren Atwood 02/08/20
//


#pragma once


#include <string>


#include "PlumeInputData.hpp"
#include "WINDSGeneralData.h"

#include "QESNetCDFOutput.h"

class Plume;

class PlumeOutput : public QESNetCDFOutput
{
public:
  // default constructor
  PlumeOutput() : QESNetCDFOutput()
  {
  }

  // specialized constructor
  PlumeOutput(PlumeInputData *PID, WINDSGeneralData *WGD, Plume *plume_ptr, std::string output_file);

  // deconstructor
  ~PlumeOutput()
  {
  }

  // setup and save output for the given time
  // in this case the saved data is output averaged concentration
  // This is the one function that needs called from outside after constructor time
  void save(float currentTime);
  void save(ptime) {}

private:
  // time averaging frequency control information
  // in this case, this is also the output control information
  float timeAvgStart;// time to start concentration averaging, not the time to start output. Adjusted if the time averaging duration does not divide evenly by the averaging frequency
  float timeAvgEnd;// time to end concentration averaging and output
  float timeAvgFreq;// time averaging frequency and output frequency


  // variables needed for getting proper averaging and output time control
  float nextOutputTime;// next output time value that is updated each time save is called and output occurs. Also initializes a restart of the particle binning for the next time averaging period

  // pointer to the class that save needs to use to get the data for the concentration calculation
  Plume *plume;


  // need nx, ny, nz of the domain to make sure the output handles domains that are not three dimensional
  // for now these are a copy of the input QES-Winds values
  int nx;
  int ny;
  int nz;

  // need the simulation timeStep for use in concentration averaging
  float timeStep;


  // Sampling box variables for calculating concentration data
  // Number of boxes to use for the sampling box
  int nBoxesX, nBoxesY, nBoxesZ;// Copies of the input: nBoxesX, Y, and Z.
  // upper & lower bounds in each direction of the sampling boxes
  float lBndx, lBndy, lBndz, uBndx, uBndy, uBndz;// Copies of the input: boxBoundsX1, boxBoundsX2, boxBoundsY1,
  float boxSizeX, boxSizeY, boxSizeZ;// these are the box sizes in each direction, calculated from nBoxes, lBnd, and uBnd variables
  float volume;// volume of the sampling boxes (=nBoxesX*nBoxesY*nBoxesZ)

  // output concentration storage variables
  std::vector<float> xBoxCen, yBoxCen, zBoxCen;// list of x,y, and z points for the concentration sampling box information
  std::vector<int> pBox;// sampling box particle counter (for average)
  std::vector<float> conc;// concentration values (for output)


  // function for counting the number of particles in the sampling boxes
  void boxCount();
};
