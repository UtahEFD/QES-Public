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
        PlumeOutputLagrToEul(PlumeInputData* PID,Dispersion* dis,std::string output_file);

        // deconstructor
        ~PlumeOutputLagrToEul()
        {
        }

        // setup and save output for the given time
        // in this case the saved data is output averaged concentration
        // This is the one function that needs called from outside after constructor time
        void save(float currentTime);

    private:

        // count number of particule in sampling box (can only be called by member)
        void boxCount(const Dispersion* disp);

        /*
            Sampling box variables for concentration data
        */

        // Copy of the input startTime, 
        // Starting time for averaging for concentration sampling
        float sCBoxTime;
        // Copy of the input timeAvg and timeStep
        float timeAvg,timeStep;
        // time of the concentration output
        float avgOutTime;
            
        // Copies of the input nBoxesX, Y, and Z. 
        // Number of boxes to use for the sampling box
        int nBoxesX,nBoxesY,nBoxesZ;    

        // Copies of the input parameters: boxBoundsX1, boxBoundsX2, boxBoundsY1, ... . 
        // upper & lower bounds in each direction of the sampling boxes
        float lBndx,lBndy,lBndz,uBndx,uBndy,uBndz;     

        // these are the box sizes in each direction
        float boxSizeX,boxSizeY,boxSizeZ;
        // volume of the sampling boxes (=nBoxesX*nBoxesY*nBoxesZ)
        float volume;      
        // list of x,y, and z points for the concentration sampling box information
        std::vector<float> xBoxCen,yBoxCen,zBoxCen;
        // sampling box particle counter (for average)
        std::vector<float> cBox;
        // concentration values (for output)
        std::vector<float> conc;      

        // pointer to the class that save needs to use to get the data for the concentration calculation
        Dispersion* disp;

};
