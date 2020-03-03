#include <iostream>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "util/ParseException.h"
#include "util/ParseInterface.h"

#include "QESNetCDFOutput.h"

#include "handleURBArgs.h"

#include "URBInputData.h"
#include "URBGeneralData.h"
#include "WINDSOutputVisualization.h"
#include "WINDSOutputWorkspace.h"

#include "TURBGeneralData.h"
#include "TURBOutput.h"

#include "Solver.h"
#include "CPUSolver.h"
#include "DynamicParallelism.h"

namespace pt = boost::property_tree;

/**
 * This function takes in a filename and attempts to open and parse it.
 * If the file can't be opened or parsed properly it throws an exception,
 * if the file is missing necessary data, an error will be thrown detailing
 * what data and where in the xml the data is missing. If the tree can't be
 * parsed, the Root* value returned is 0, which will register as false if tested.
 * @param fileName the path/name of the file to be opened, must be an xml
 * @return A pointer to a root that is filled with data parsed from the tree
 */
URBInputData* parseXMLTree(const std::string fileName);

int main(int argc, char *argv[])
{
    // CUDA-Urb - Version output information
    std::string Revision = "0";
    std::cout << "cudaUrb " << "0.8.0" << std::endl;
    
    // ///////////////////////////////////
    // Parse Command Line arguments
    // ///////////////////////////////////
    
    // Command line arguments are processed in a uniform manner using
    // cross-platform code.  Check the URBArgs class for details on
    // how to extend the arguments.
    URBArgs arguments;
    arguments.processArguments(argc, argv);
    
    // ///////////////////////////////////
    // Read and Process any Input for the system
    // ///////////////////////////////////
    
    // Parse the base XML QUIC file -- contains simulation parameters
    URBInputData* UID = parseXMLTree(arguments.quicFile);
    if ( !UID ) {
        std::cerr << "QUIC Input file: " << arguments.quicFile <<
            " not able to be read successfully." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    // Generate the general URB data from all inputs
    // - turn on mixing length calculations for now if enabled
    // eventually, we will remove this flag and 2nd argument
    if (arguments.calcMixingLength) {
        std::cout << "Enabling mixing length calculation." << std::endl;
    }
    URBGeneralData* UGD = new URBGeneralData(UID, arguments.calcMixingLength);
    
    // create URB output classes
    std::vector<QESNetCDFOutput*> outputVec;
    if (arguments.netCDFFileVz != "") {
        outputVec.push_back(new WINDSOutputVisualization(UGD,UID,arguments.netCDFFileVz));
    }
    if (arguments.netCDFFileWk != "") {
        outputVec.push_back(new WINDSOutputWorkspace(UGD,arguments.netCDFFileWk));
    }
    
    
    // Generate the general TURB data from URB data
    // based on if the turbulence output file is defined
    TURBGeneralData* TGD = nullptr;
    if (arguments.netCDFFileTurb != "") {
        TGD = new TURBGeneralData(UGD);
        outputVec.push_back(new TURBOutput(TGD,arguments.netCDFFileTurb));
    }
    
    // //////////////////////////////////////////
    //
    // Run the CUDA-URB Solver
    //
    // //////////////////////////////////////////
    Solver *solver, *solverC = nullptr;
    if (arguments.solveType == CPU_Type) {
        std::cout << "Run CPU Solver ..." << std::endl;
        solver = new CPUSolver(UID, UGD);
    } else if (arguments.solveType == DYNAMIC_P) {
        std::cout << "Run GPU Solver ..." << std::endl;
        solver = new DynamicParallelism(UID, UGD);
    } else {
        std::cerr << "Error: invalid solve type\n";
        exit(EXIT_FAILURE);
    }
    
    //check for comparison
    if (arguments.compareType) {
        if (arguments.compareType == CPU_Type)
            solverC = new CPUSolver(UID, UGD);
        else if (arguments.compareType == DYNAMIC_P)
            solverC = new DynamicParallelism(UID, UGD);
        else {
            std::cerr << "Error: invalid comparison type\n";
            exit(EXIT_FAILURE);
        }
    }
    
    // Run urb simulation code
    solver->solve(UID, UGD, !arguments.solveWind );
    
    std::cout << "Solver done!\n";
    
    if (solverC != nullptr) {
        std::cout << "Running comparson type...\n";
        solverC->solve(UID, UGD, !arguments.solveWind);
    }
    
    // /////////////////////////////
    //
    // Run turbulence
    //
    // /////////////////////////////
    if(TGD != nullptr) {
        TGD->run(UGD);
    }

    // /////////////////////////////
    // Output the various files requested from the simulation run
    // (netcdf wind velocity, icell values, etc...
    // /////////////////////////////
    for(auto id_out=0u;id_out<outputVec.size();id_out++)
        outputVec.at(id_out)->save(0.0); // need to replace 0.0 with timestep
    
    
    exit(EXIT_SUCCESS);
}

URBInputData* parseXMLTree(const std::string fileName)
{
	pt::ptree tree;

	try
	{
		pt::read_xml(fileName, tree);
	}
	catch (boost::property_tree::xml_parser::xml_parser_error& e)
	{
		std::cerr << "Error reading tree in" << fileName << "\n";
		return (URBInputData*)0;
	}

	URBInputData* xmlRoot = new URBInputData();
        xmlRoot->parseTree( tree );
	return xmlRoot;
}
