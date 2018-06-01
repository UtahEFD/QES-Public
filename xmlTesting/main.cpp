#include "ParseInterface.h"
#include "X.h"
#include "A.h"
#include "B.h"
#include "C.h"

#include <string>
#include <iostream>
#include <iomanip>

#include <vector>

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

	ParseInterface structureParsed;

	structureParsed.parseValues(tree);

	std::cout << "X" << std::endl << structureParsed.xVar->x << std::endl;
	std::cout << "A" << std::endl;
	for (int i = 0; i < structureParsed.xVar->aVar->vals.size(); i++)
		std::cout << structureParsed.xVar->aVar->vals[i] << std::endl;
	std::cout << "B" << std::endl;
	std::cout << structureParsed.xVar->bVar->numAs << std::endl;
	for (int i = 0; i < structureParsed.xVar->bVar->aVals.size(); i++)
	{
		std::cout << "A" << std::endl;
		for (int j = 0; j < (structureParsed.xVar->bVar->aVals[i])->vals.size(); j++)
			std::cout << (structureParsed.xVar->bVar->aVals[i])->vals[j] << std::endl;
	}
	std::cout << "C" << std::endl;
	std::cout << structureParsed.xVar->cVar->x << std::endl << structureParsed.xVar->cVar->y << std::endl;


}