#include <iostream>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "util/ParseException.h"
#include "util/ParseInterface.h"

#include "handleWINDSArgs.h"

#include "WINDSInputData.h"
#include "WINDSGeneralData.h"

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
WINDSInputData* parseXMLTree(const std::string fileName);

int main(int argc, char *argv[])
{

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

    // Generate the general WINDS data from all inputs
    WINDSGeneralData* WGD = new WINDSGeneralData(WID);
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
