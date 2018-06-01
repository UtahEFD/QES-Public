#include "ParseInterface.h"
#include "A.h"
#include "B.h"
#include "C.h"

#include <string>
#include <iostream>
#include <iomanip>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;

int main()
{
	std::string fileName, logFileName;
	int logLevel;
	std::cout << "Input the file name: ";
	std::cin >> fileName;

	pt::ptree tree;

	pt::read_xml(fileName, tree);

	logFileName = tree.get<std::string>("debug.filename");

	logLevel = tree.get("debug.level", 0);

	BOOST_FOREACH(pt::ptree::value_type const &v, tree.get_child("X")){

	}


}