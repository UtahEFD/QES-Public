//
//  NetCDFOutputEulerian.h
//  
//  This class handles saving output files for input Eulerian data
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
#include "Turb.hpp"
#include "Eulerian.h"


#include "NetCDFOutputGeneric.h"


class PlumeOutputEulerian : public NetCDFOutputGeneric
{
    public:

        // default constructor
        PlumeOutputEulerian():NetCDFOutputGeneric()
        {
        }

        // specialized constructor
        PlumeOutputEulerian(PlumeInputData* PID,Urb* urb_ptr,Turb* turb_ptr,Eulerian* eul_ptr,std::string output_file);

        // deconstructor
        ~PlumeOutputEulerian()
        {
        }

        // setup and save output for the given time
        // in this case the saved data is output averaged concentration
        // This is the one function that needs called from outside after constructor time
        void save(float currentTime);

    private:

        
        // no need for output frequency for this output, it is expected to only happen once, assumed to be at time zero

        // pointers to the classes that save needs to use to get the data for the output
        Urb* urb;
        Turb* turb;
        Eulerian* eul;

        // main output metadata
        // LA future work: this whole structure will have to change when we finally adjust the inputs for the true grids
        //  would mean cell centered urb data and face centered turb data. For now, decided just to assume they have the same grid
        int nx;
        int ny;
        int nz;
        int nCells;
        
        // other output data
        std::vector<float> epps;    // data is normally stored as CoEps, so need to separate it out here

};