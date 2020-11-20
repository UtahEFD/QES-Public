
#include <iostream>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>

#include "util/ParseException.h"
#include "util/ParseInterface.h"

#include "handleWINDSArgs.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

#include "Solver.h"
#include "CPUSolver.h"
#include "DynamicParallelism.h"

#include "Fire.hpp"
#include "FIREOutput.h"
//#include "DTEHeightField.h"



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
WINDSInputData* parseXMLTree(const std::string fileName);

int main(int argc, char *argv[])
{
    std::string Revision = "0";
    // CUDA-WINDS - Version output information
    std::cout << "cudaWINDS " << "0.8.0" << std::endl;

    // ///////////////////////////////////
    // Parse Command Line arguments
    // ///////////////////////////////////

    // Command line arguments are processed in a uniform manner using
    // cross-platform code.  Check the WINDSArgs class for details on
    // how to extend the arguments.
    WINDSArgs arguments;
    arguments.processArguments(argc, argv);

    // ///////////////////////////////////
    // Read and Process any Input for the system
    // ///////////////////////////////////

    // Parse the base XML QUIC file -- contains simulation parameters
    WINDSInputData* WID = parseXMLTree(arguments.quicFile);
    if ( !WID ) {
        std::cerr << "QUIC Input file: " << arguments.quicFile << " not able to be read successfully." << std::endl;
        exit(EXIT_FAILURE);
    }

    //if the commandline dem file is blank, and a file was specified in the xml,
    //use the dem file from the xml
    /*
    std::string demFile = "";
    if (arguments.demFile != "")
        demFile = arguments.demFile;
    else if (WID->simParams && WID->simParams->demFile != "")
        demFile = WID->simParams->demFile;


    DTEHeightField* DTEHF = 0;
    if (demFile != "") {
        DTEHF = new DTEHeightField(demFile, (*(WID->simParams->grid))[0], (*(WID->simParams->grid))[1] );
    }

    if (DTEHF) {
        std::cout << "Forming triangle mesh...\n";
        DTEHF->setDomain(WID->simParams->domain, WID->simParams->grid);
        std::cout << "Mesh complete\n";
    }

    if (arguments.terrainOut) {
        if (DTEHF) {
            std::cout << "Creating terrain OBJ....\n";
            DTEHF->outputOBJ("terrain.obj");
            std::cout << "OBJ created....\n";
        }
        else {
            std::cerr << "Error: No dem file specified as input\n";
            return -1;
        }
    }
    */

    // Files was successfully read, so create instance of output class
    /*
      Output* output = nullptr;
    if (WID->fileOptions->outputFlag==1) {
        output = new Output(arguments.netCDFFile);
    }
    */


    // Generate the general WINDS data from all inputs
    WINDSGeneralData* WGD = new WINDSGeneralData(WID);
    
    
    // //////////////////////////////////////////
    //
    // Run the CUDA-WINDS Solver
    //
    // //////////////////////////////////////////
    Solver *solver, *solverC = nullptr;
    if (arguments.solveType == CPU_Type)
        solver = new CPUSolver(WID, WGD);
    else if (arguments.solveType == DYNAMIC_P)
        solver = new DynamicParallelism(WID, WGD);
    else
    {
        std::cerr << "Error: invalid solve type\n";
        exit(EXIT_FAILURE);
    }

    //check for comparison
    if (arguments.compareType)
    {
        if (arguments.compareType == CPU_Type)
            solverC = new CPUSolver(WID, WGD);
        else if (arguments.compareType == DYNAMIC_P)
            solverC = new DynamicParallelism(WID, WGD);
        else
        {
            std::cerr << "Error: invalid comparison type\n";
            exit(EXIT_FAILURE);
        }
    }
    /*
    //close the scanner
    if (DTEHF)
        DTEHF->closeScanner();
    */
    // save initial fields to reset after each time+fire loop
    std::vector<float> u0 = WGD->u0;
    std::vector<float> v0 = WGD->v0;
    std::vector<float> w0 = WGD->w0;
    
    /** 
     * Create Fire Mapper
     **/
    Fire* fire = new Fire(WID, WGD);
    
    // create FIREOutput manager
    FIREOutput* fireOutput = new FIREOutput(WGD,fire,arguments.netCDFFileFire);
    
    // set base w in fire model to initial w0
    //fire->w_base = w0;
    
    // Run initial solver to generate full field
    solver->solve(WID, WGD, !arguments.solveWind);
    
    // save initial fields in solver and fire
    //if (output != nullptr) {
    //    WGD->save();
    //}
    
    // save any fire data (at time 0)
    fireOutput->save(0.0);
	
    // Run WINDS simulation code
    std::cout<<"===================="<<std::endl;
    double t = 0;
    
    while (t<WID->simParams->totalTimeIncrements) {
        
        std::cout<<"Processing time t = "<<t<<std::endl;
        // re-set initial fields after first time step
        if (t>0) {

	    
	    WGD->u0 = u0;
	    WGD->v0 = v0;
	    WGD->w0 = w0;
	    
	    /*
	    WID->metParams->z0_domain_flag=1;
	    WID->metParams->sensors[0]->inputWindProfile(WID, WGD);
            solver->solve(WID, WGD, !arguments.solveWind);
	    */
        }
        
        // loop 2 times for fire
        int loop = 0;
        while (loop<1) {
            
            // run Balbi model to get new spread rate and fire properties
            fire->run(solver, WGD);
            
	    // calculate plume potential
	    auto start = std::chrono::high_resolution_clock::now(); // Start recording execution time

	    fire->potential(WGD);

	    auto finish = std::chrono::high_resolution_clock::now();  // Finish recording execution time

    	    std::chrono::duration<float> elapsed = finish - start;
    	    std::cout << "Plume solve: elapsed time: " << elapsed.count() << " s\n";   // Print out elapsed execution time
	  
	    // run wind solver
            solver->solve(WID, WGD, !arguments.solveWind);
	    
            //increment fire loop
            loop += 1;
            
            std::cout<<"--------------------"<<std::endl;
        }
                
        // move the fire
        fire->move(solver, WGD);
                
        // save solver data
        //if (output != nullptr) {
        //    WGD->save();
        //}
        
        // save any fire data
        fireOutput->save(fire->time);

        // advance time 
        t = fire->time;
        

    }

    
    exit(EXIT_SUCCESS);
}

WINDSInputData* parseXMLTree(const std::string fileName)
{
	pt::ptree tree;

	try
	{
		pt::read_xml(fileName, tree);
	}
	catch (boost::property_tree::xml_parser::xml_parser_error& e)
	{
		std::cerr << "Error reading tree in" << fileName << "\n";
		return (WINDSInputData*)0;
	}

	WINDSInputData* xmlRoot = new WINDSInputData();
        xmlRoot->parseTree( tree );
	return xmlRoot;
}
