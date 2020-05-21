//
//  NetCDFOutputEulerian.h
//  
//  This class handles saving output files for input Eulerian data
//  This is a specialized output class derived 
//   and inheriting from QESNetCDFOutput.
//
//  Created by Fabien Margairaz on 01/25/20
//  Modified by Loren Atwood 02/08/20
//

#pragma once


#include <string>


#include "PlumeInputData.hpp"
#include "URBGeneralData.h"
#include "TURBGeneralData.h"
#include "Eulerian.h"

#include "QESNetCDFOutput.h"


class PlumeOutputEulerian : public QESNetCDFOutput
{
public:
    
    // default constructor
    PlumeOutputEulerian():QESNetCDFOutput()
    {
    }
    
    // specialized constructor
    PlumeOutputEulerian(PlumeInputData*,URBGeneralData*,TURBGeneralData*,Eulerian*,std::string);
    
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
    URBGeneralData* urb_;
    TURBGeneralData* turb_;
    Eulerian* eul_;
    
    // other output data
    std::vector<float> epps;    // data is normally stored as CoEps, so need to separate it out here
    
};
