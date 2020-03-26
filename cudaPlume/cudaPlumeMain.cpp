
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
#include "Plume.hpp"
#include "Eulerian.h"
#include "Dispersion.h"


#include "NetCDFOutputGeneric.h"
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


// leave this as 0 if you are using input urb and turb files from current urb and turb
// set this to 1 if you are using input urb and turb files from past urb and turb
// LA future work: the Bailey test cases are currently using past urb and turb file formats. Work needs done to 
//  update their format to the current urb and turb formats, then we can just use one input constructor.
//  I'm waiting to do this because I want to figure out why there are still rogue trajectories for the LES case before I complicate
//  things by varying the file formats. Changing formats for the Bailey test cases would require a grid shift, 
//  meaning careful placement and choice for an extra value in each dimension.
#define USE_PREVIOUSCODE 0


#if USE_PREVIOUSCODE

    // Create instance of cudaUrb input class
    Input* inputUrb = new Input(arguments.inputUrbFile);
    
    // Create instance of cudaTurb input class
    Input* inputTurb = new Input(arguments.inputTurbFile);

#else

    // Create instance of cudaUrb input class
    NetCDFInput* inputUrb = new NetCDFInput(arguments.inputUrbFile);

    // Create instance of cudaTurb input class
    NetCDFInput* inputTurb = new NetCDFInput(arguments.inputTurbFile);

#endif

    
    // Create instance of cudaUrb class
    Urb* urb = new Urb(inputUrb, arguments.debug);
    
    // Create instance of cudaTurb class
    Turb* turb = new Turb(inputTurb, arguments.debug);
    
    // Create instance of Eulerian class
    Eulerian* eul = new Eulerian(PID,urb,turb, arguments.debug);
    
    // Create instance of Dispersion class
    Dispersion* dis = new Dispersion(PID,urb,turb,eul, arguments.debug);
    


    // create output instance
    // LA note: start it out as NULL, then make it point to what we want later if the file is supposed to exist
    PlumeOutputEulerian* eulOutput = NULL;
    PlumeOutputLagrToEul* lagrToEulOutput = NULL;
    PlumeOutputLagrangian* lagrOutput = NULL;
    if( arguments.doEulDataOutput == true )
    {
        eulOutput = new PlumeOutputEulerian(PID,urb,turb,eul,arguments.outputEulerianFile);
    }
    // always supposed to output lagrToEulOutput data
    lagrToEulOutput = new PlumeOutputLagrToEul(PID,urb,dis,arguments.outputLagrToEulFile);
    if( arguments.doLagrDataOutput == true )
    {
        lagrOutput = new PlumeOutputLagrangian(PID,dis,arguments.outputLagrangianFile);
    }

    // output Eulerian data. Use time zero
    if( arguments.doEulDataOutput == true )
    {
        eulOutput->save(0.0);
    }

    
    // Create instance of Plume model class
    Plume* plume = new Plume(PID,urb,turb,eul,dis, arguments.doLagrDataOutput, arguments.doSimInfoFileOutput,arguments.outputFolder,arguments.caseBaseName, arguments.debug);
    
    // Run plume advection model
    plume->run(lagrToEulOutput,lagrOutput);
    
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
 
