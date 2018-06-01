#pragma once

#include <string>

#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/optional.hpp>
#include <iostream>

#include <vector>

class Root;

namespace pt = boost::property_tree;

/**
 * This class is a generic object from which all classes that can be parsed from an XML 
 * file will inherit from. This class contains methods to simplify the parsing process
 * and obscure the complecations of the boost library.
 */

class ParseInterface
{
private:

protected:
	pt::ptree tree;

	 /**
	 * This sets the tree of this object
	 */
	void setTree(pt::ptree t) { tree = t;}

public:
	Root* root;
	/**
	 * default constructor
	 */
	ParseInterface();

	/**
	 * sets root to null, takes in a default tree
	 */
	ParseInterface(pt::ptree t);

	/**
	 * This function parses the current node of the tree and searches for an element
	 * with the tag of "tag". Once it is found, it returns the value.
	 * this is a template function, so the type of the value that is taken in and
	 * found are flexible. If the value is unable to be found, "val" will not be
	 * updated.
	 * @param val the value that is being updated
	 * @param tag the tagline in the xml of the value we are searching for
	 */
	template <typename T>
	void parsePrimative(T& val, const std::string tag);

	/**
	 * This function parses the current node of the tree and searches for all elements
	 * with the tag of "tag". When one is found, it adds the value to a vector.
	 * this is a template function, so the type of the value that is taken in and
	 * found are flexible. This will return all instances of a certain tag, this means it
	 * can have 0..* values.
	 * @param vals the vector to be filled with found values
	 * @param tag the tagline in the xml of the values we are searching for
	 */
	template <typename T>
	void parseMultiPrimatives(std::vector<T>& vals, const std::string tag);


	/**
	 * This function parses the current node of the tree and searches for an element
	 * with the tag of "tag". Once it is found, that element will be 
	 * parsed and the completed element will be returned.
	 * this is a template function, so the type of the value that is taken in and
	 * found are flexible. If the value is unable to be found, "val" will not be
	 * updated.
	 * @param ele the element that is being parsed and updated
	 * @param tag the tagline in the xml of the element we are searching for
	 */
	template <typename T>
	void parseElement(T*& ele, const std::string tag);


	/**
	 * This function parses the current node of the tree and searches for all elements
	 * with the tag of "tag". Once it is found, that element will be 
	 * parsed and the completed element will be returned.
	 * this is a template function, so the type of the value that is taken in and
	 * found are flexible. This will return all instances of a certain tag, this means it
	 * can have 0..* elements.
	 * @param eles the vector to be filled with found elements
	 * @param tag the tagline in the xml of the elements we are searching for
	 */
	template <typename T>
	void parseMultiElements(std::vector<T*>& eles, const std::string tag);


	/**
	 * Virtual function
	 * This is where we select what values we want to parse out of
	 * the xml file by calling the parseX functions above.
	 */
	virtual void parseValues();

	/**
	 * This function returns the root value
	 */
	Root* getRoot() { return root; }


};