#pragma once

#include <string>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/optional.hpp>

#include <vector>

class X;

namespace pt = boost::property_tree;

class ParseInterface
{
public:
	X* xVar;

	ParseInterface();

	template <typename T>
	void parsePrimative(T& val,const std::string tag, const pt::ptree tree);

	template <typename T>
	void parseMultiPrimatives(std::vector<T>& vals, const std::string tag, const pt::ptree tree);

	template <typename T>
	void parseElement(T*& ele, const std::string tag, const pt::ptree tree);

	template <typename T>
	void parseMultiElements(std::vector<T*>& eles, const std::string tag, const pt::ptree tree);


	virtual void parseValues(const pt::ptree tree);


};