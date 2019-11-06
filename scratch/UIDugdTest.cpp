#include <iostream>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "util/ParseException.h"
#include "util/ParseInterface.h"

#include "handleURBArgs.h"

#include "URBInputData.h"
#include "URBGeneralData.h"

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

    // Generate the general URB data from all inputs
    URBGeneralData* UGD = new URBGeneralData(UID);
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
