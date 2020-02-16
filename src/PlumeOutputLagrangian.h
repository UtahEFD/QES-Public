//
//  NetCDFOutputLagrangian.h
//  
//  This class handles saving output files for Lagrangian particle data
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



class PlumeOutputLagrangian : public NetCDFOutputGeneric
{
    public:

        // default constructor
        PlumeOutputLagrangian() : NetCDFOutputGeneric()
        {
        }

        // specialized constructor
        PlumeOutputLagrangian(PlumeInputData* PID,Dispersion* dis_ptr,std::string output_file);

        // deconstructor
        ~PlumeOutputLagrangian()	       
        {
        }

        // setup and save output for the given time
        // in this case the saved data is output averaged concentration
        // This is the one function that needs called from outside after constructor time
        void save(float);

    private:

        // Output frequency control information
        // FM -> need to create dedicated input variables
        float outputStartTime;   // time to start output
        float outputEndTime;  // time to end output
        float outputFrequency;  // output frequency

        // next output time value that is updated each time save is called and there is output
        float nextOutputTime;

        // pointer to the class that save needs to use to get the data for the concentration calculation
        Dispersion* disp;


        // main particle metadata, almost a copy of what is in the "particle" class
        int numPar;     // total number of particle to be released 
        std::vector<int> parID;     // list of particle IDs (for NetCDF dimension)
        std::vector<float> xPos_init;   // list of initial x positions for the particles
        std::vector<float> yPos_init;   // list of initial y positions for the particles
        std::vector<float> zPos_init;   // list of initial z positions for the particles
        std::vector<float> tStrt;       // list of release times for the particles
        std::vector<int> sourceIdx;     // list of sourceIdx for the particles

        // other particle data, definitely a copy of what is in the "particle" class,
        // but only the particle information that matters for particle statistic calculations
        // and particle info plotting
        std::vector<float> xPos;        // list of x positions for the particles
        std::vector<float> yPos;        // list of y positions for the particles
        std::vector<float> zPos;        // list of z positions for the particles
        std::vector<float> uFluct;      // list of u velocity fluctuations for the particles
        std::vector<float> vFluct;      // list of v velocity fluctuations for the particles
        std::vector<float> wFluct;      // list of w velocity fluctuations for the particles
        std::vector<float> delta_uFluct;    // list of the uFluct differences for the particles
        std::vector<float> delta_vFluct;    // list of the vFluct differences for the particles
        std::vector<float> delta_wFluct;    // list of the wFluct differences for the particles
        std::vector<int> isRogue;          // list of isRogue info for the particles
        std::vector<int> isActive;         // // list of isActive info for the particles
        
};
