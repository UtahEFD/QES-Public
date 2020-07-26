#pragma once

/*
 * This class is a container relating to sensors and metric
 * information read from the xml.
 */
#include <algorithm>

#include "util/ParseInterface.h"
#include "util/ParseException.h"
#include "Sensor.h"
#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <string>

namespace pt = boost::property_tree;

class MetParams : public ParseInterface
{
private:



public:

	int z0_domain_flag = 0;
	std::vector<Sensor*> sensors;

	std::vector<std::string> sensorName;



	virtual void parseValues()
	{
		parsePrimitive<int>(false, z0_domain_flag, "z0_domain_flag");
		parseMultiElements<Sensor>(false, sensors, "sensor");

		parseMultiPrimitives<std::string>(false, sensorName, "sensorName");

	}

	


};
