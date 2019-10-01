#include <iostream>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "util/ParseException.h"
#include "util/ParseInterface.h"

#include "handleURBArgs.h"

#include "URBInputData.h"
#include "URBGeneralData.h"
#include "URBOutput_Static.h"
#include "URBOutput_WindVelCellCentered.h"
#include "URBOutput_WindVelFaceCentered.h"

#include "Solver.h"
#include "CPUSolver.h"
#include "DynamicParallelism.h"
#include "Output.hpp"

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
        std::cerr << "QUIC Input file: " << arguments.quicFile << " not able to be read successfully." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Files was successfully read, so create instance of output class
    Output* output = nullptr;
    if (UID->fileOptions->outputFlag==1) {
      output = new Output(arguments.netCDFFile);
    }
    
    // Generate the general URB data from all inputs
    URBGeneralData* UGD = new URBGeneralData(UID, output);

    // create URB output  
    URBOutput_Static* output_st = nullptr;
    if (UID->fileOptions->outputFlag==1) {
      std::string fname=arguments.netCDFFile;
      fname.replace(fname.end()-3,fname.end(),"_st.nc");
      output_st = new URBOutput_Static(UGD,fname);
      output_st->save(UGD);
    }
    URBOutput_WindVelCellCentered* output_cc = nullptr;
    if (UID->fileOptions->outputFlag==1) {
      std::string fname=arguments.netCDFFile;
      fname.replace(fname.end()-3,fname.end(),"_cc.nc");
      output_cc = new URBOutput_WindVelCellCentered(UGD,fname);
    }
    URBOutput_WindVelFaceCentered* output_fc = nullptr;
    if (UID->fileOptions->outputFlag==1) {
      std::string fname=arguments.netCDFFile;
      fname.replace(fname.end()-3,fname.end(),"_fc.nc");
      output_fc = new URBOutput_WindVelFaceCentered(UGD,fname);
    }
    
    // //////////////////////////////////////////
    //
    // Run the CUDA-URB Solver
    //
    // //////////////////////////////////////////
    Solver *solver, *solverC = nullptr;
    if (arguments.solveType == CPU_Type)
        solver = new CPUSolver(UID, UGD);
    else if (arguments.solveType == DYNAMIC_P)
        solver = new DynamicParallelism(UID, UGD);
    else
    {
        std::cerr << "Error: invalid solve type\n";
        exit(EXIT_FAILURE);
    }

    //check for comparison
    if (arguments.compareType)
    {
        if (arguments.compareType == CPU_Type)
            solverC = new CPUSolver(UID, UGD);
        else if (arguments.compareType == DYNAMIC_P)
            solverC = new DynamicParallelism(UID, UGD);
        else
        {
            std::cerr << "Error: invalid comparison type\n";
            exit(EXIT_FAILURE);
        }
    }
    // Run urb simulation code
    solver->solve(UID, UGD, !arguments.solveWind );

    std::cout << "Solver done!\n";

    if (solverC != nullptr)
    {
        std::cout << "Running comparson type...\n";
        solverC->solve(UID, UGD, !arguments.solveWind);
    }

    // /////////////////////////////
    // Output the various files requested from the simulation run
    // (netcdf wind velocity, icell values, etc...
    // /////////////////////////////
    if (output) {
      UGD->save();
    }
    if (output_cc) {
      output_cc->save(UGD);
    }
    if (output_fc) {
      //output_fc->save(UGD);
    }

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
