
#include <iostream>
#include <netcdf>
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>


#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "util/calcTime.h"


#include "Args.hpp"
#include "PlumeInputData.hpp"
#include "Input.hpp"
#include "NetCDFInput.h"
#include "Urb.hpp"
#include "Turb.hpp"

#include "URBGeneralData.h"
#include "TURBGeneralData.h"

#include "Plume.hpp"
#include "Eulerian.h"
#include "Dispersion.h"

#include "QESNetCDFOutput.h"
#include "PlumeOutputEulerian.h"
#include "PlumeOutputLagrToEul.h"
#include "PlumeOutputLagrangian.h"


// LA do these need to be here???
using namespace netCDF;
using namespace netCDF::exceptions;

PlumeInputData* parseXMLTree(const std::string fileName);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
    // set up timer information for the simulation runtime
    calcTime timers;
    timers.startNewTimer("CUDA-Plume total runtime"); // start recording execution time

    
    // print a nice little welcome message
    std::cout << std::endl;
    std::cout<<"##############################################################"<<std::endl;
    std::cout<<"#                                                            #"<<std::endl;
    std::cout<<"#                   Welcome to CUDA-PLUME                    #"<<std::endl;
    std::cout<<"#                                                            #"<<std::endl;
    std::cout<<"##############################################################"<<std::endl;
    
    // parse command line arguments
    Args arguments;
    arguments.processArguments(argc, argv);
    
    // parse xml settings
    PlumeInputData* PID = parseXMLTree(arguments.inputQuicFile);
    if ( !PID ) {
        std::cerr << "QUIC-Plume input file: " << arguments.inputQuicFile << " not able to be read successfully." << std::endl;
        exit(EXIT_FAILURE);
    }

    
    // Create instance of QES-winds General data class
    URBGeneralData* UGD = new URBGeneralData(&arguments);
    //load data at t=0;
    UGD->loadNetCDFData(0);
    
    // Create instance of QES-Turb General data class
    TURBGeneralData* TGD = new TURBGeneralData(&arguments,UGD);
    //load data at t=0;
    TGD->loadNetCDFData(0);

    // Create instance of Eulerian class
    Eulerian* eul = new Eulerian(PID,UGD,TGD, arguments.debug);
    
    // Create instance of Dispersion class
    //Dispersion* dis = new Dispersion(PID,UGD,TGD,eul, arguments.debug);
    
    // Create instance of Plume model class
    Plume* plume = new Plume(PID,UGD,TGD,eul,&arguments);
    
    // create output instance
    std::vector<QESNetCDFOutput*> outputVec;
    // always supposed to output lagrToEulOutput data
    outputVec.push_back(new PlumeOutputLagrToEul(PID,UGD,plume,arguments.outputLagrToEulFile));
    if( arguments.doLagrDataOutput == true ) {
        outputVec.push_back(new PlumeOutputLagrangian(PID,plume,arguments.outputLagrangianFile));
    }
    
    // create output instance (separate for eulerian class)
    QESNetCDFOutput* eulOutput = nullptr;
    if( arguments.doEulDataOutput == true ) {
        eulOutput = new PlumeOutputEulerian(PID,UGD,TGD,eul,arguments.outputEulerianFile);
        // output Eulerian data. Use time zero
        eulOutput->save(0.0);
    }
    
    // Run plume advection model
    plume->run(UGD,TGD,eul,outputVec);
    
    // compute run time information and print the elapsed execution time
    std::cout<<"[CUDA-Plume] \t Finished."<<std::endl;
    timers.printStoredTime("CUDA-Plume total runtime");
    std::cout<<"##############################################################"<<std::endl;
    
    exit(EXIT_SUCCESS);
  
}

PlumeInputData* parseXMLTree(const std::string fileName)
{
	pt::ptree tree;

	try
	{
		pt::read_xml(fileName, tree);
	}
	catch (boost::property_tree::xml_parser::xml_parser_error& e)
	{
		std::cerr << "Error reading tree in" << fileName << "\n";
		return (PlumeInputData*)0;
	}

	PlumeInputData* xmlRoot = new PlumeInputData();
        xmlRoot->parseTree( tree );
	return xmlRoot;
}
 
