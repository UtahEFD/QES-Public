//
//  NetCDFOutputLagrToEul.h
//  
//  This class handles saving output files for Eulerian binned Lagrangian particle data,
//   where this class handles the binning of the Lagrangian particle data
//  This is a specialized output class derived 
//   and inheriting from NetCDFOutputGeneric.
//
//  Created by Fabien Margairaz on 01/25/20
//  Modified by Loren Atwood 02/08/20
//


#pragma once


#include <string>


#include "PlumeInputData.hpp"
#include "Urb.hpp"
#include "Dispersion.h"


#include "NetCDFOutputGeneric.h"


class PlumeOutputLagrToEul : public NetCDFOutputGeneric
{
    public:

        // default constructor
        PlumeOutputLagrToEul():NetCDFOutputGeneric()
        {
        }

        // specialized constructor
        PlumeOutputLagrToEul(PlumeInputData* PID,Urb* urb_ptr,Dispersion* dis_ptr,std::string output_file);

        // deconstructor
        ~PlumeOutputLagrToEul()
        {
        }

        // setup and save output for the given time
        // in this case the saved data is output averaged concentration
        // This is the one function that needs called from outside after constructor time
        void save(float currentTime);

    private:

        // Output frequency control information
        // in this case, this is also the averaging control information
        float timeAvgStart;     // time to start concentration averaging and output
        float timeAvgEnd;       // time to end concentration averaging and output
        float timeAvgFreq;      // time averaging frequency and output frequency
        

        // next averaging time value that is updated each time save is called and averaging occurs
        // is also the next output time value
        float nextAvgTime;
        
        // pointer to the class that save needs to use to get the data for the concentration calculation
        Dispersion* disp;


        // need nx, ny, nz of the domain to make sure the output handles domains that are not three dimensional
        // for now these are a copy of the input urb values
        int nx;
        int ny;
        int nz;

        // need the simulation timeStep for use in concentration averaging
        float timeStep;


        // Sampling box variables for calculating concentration data
        int nBoxesX,nBoxesY,nBoxesZ;    // Copies of the input nBoxesX, Y, and Z. // Number of boxes to use for the sampling box
        float lBndx,lBndy,lBndz,uBndx,uBndy,uBndz;  // Copies of the input parameters: boxBoundsX1, boxBoundsX2, boxBoundsY1, upper & lower bounds in each direction of the sampling boxes
        float boxSizeX,boxSizeY,boxSizeZ;   // these are the box sizes in each direction, calculated from nBoxes, lBnd, and uBnd variables
        float volume;   // volume of the sampling boxes (=nBoxesX*nBoxesY*nBoxesZ)
        
        // output concentration storage variables
        std::vector<float> xBoxCen,yBoxCen,zBoxCen;     // list of x,y, and z points for the concentration sampling box information
        std::vector<float> cBox;    // sampling box particle counter (for average)
        std::vector<float> conc;    // concentration values (for output)


        // function for counting the number of particles in the sampling boxes
        void boxCount();

};
