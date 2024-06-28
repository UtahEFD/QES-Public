#pragma once

#include <string>

#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/optional.hpp>
#include <iostream>

#include <vector>
#include "ParseException.h"

// class URBInputData;

namespace pt = boost::property_tree;

/**
 * This structure is for grouping information necessary
 * for parsing out polymorphic objects from an XML. This takes
 * two template types, one for the base class, and one for the
 * inheriting class. The constructor takes in a string which
 * is used as a tag for finding values in the XML
 */

template<typename generic, typename thisType>
struct Polymorph
{
  typedef thisType T;
  typedef generic G;
  std::string tag;

  Polymorph(std::string t)
  {
    tag = t;
  }

  void setNewType(G *&element)
  {
    element = new T();
  }
};


/**
 * This class is a generic object from which all classes that can be parsed from an XML
 * file will inherit from. This class contains methods to simplify the parsing process
 * and obscure the complecations of the boost library.
 */

class ParseInterface
{
private:
protected:
  pt::ptree tree;// XML tree at this point in parsing
  std::string treeParents;// a string listing the parents' tags in order

  /**
   * This sets the tree of this object
   * @param t tree to be set
   */
  void setTree(pt::ptree t) { tree = t; }

  /**
   * sets the string of parent tags
   * @param s string of parent tags
   */
  void setParents(std::string s) { treeParents = s; }

public:
  /**
   * This function parses the current node of the tree and searches for an element
   * with the tag of "tag". Once it is found, it returns the value.
   * this is a template function, so the type of the value that is taken in and
   * found are flexible. If the value is unable to be found, "val" will not be
   * updated.
   * If this variable is required and not found, an exception will be thrown.
   * @param isReq if this variable is required or not
   * @param val the value that is being updated
   * @param tag the tagline in the xml of the value we are searching for
   */
  template<typename T>
  void parsePrimitive(bool isReq, T &val, const std::string tag);

  /**
   * This function parses the current node of the tree and searches for all elements
   * with the tag of "tag". When one is found, it adds the value to a vector.
   * this is a template function, so the type of the value that is taken in and
   * found are flexible. This will return all instances of a certain tag, this means it
   * can have 0..* values.
   * If this variable is required and not found, an exception will be thrown.
   * @param isReq if this variable is required or not
   * @param vals the vector to be filled with found values
   * @param tag the tagline in the xml of the values we are searching for
   */
  template<typename T>
  void parseMultiPrimitives(bool isReq, std::vector<T> &vals, const std::string tag);


  /**
   * This function parses the current node of the tree and searches for an element
   * with the tag of "tag". Once it is found, that element will be
   * parsed and the completed element will be returned.
   * this is a template function, so the type of the value that is taken in and
   * found are flexible. If the value is unable to be found, "val" will not be
   * updated.
   * If this variable is required and not found, an exception will be thrown.
   * @param isReq if this variable is required or not
   * @param ele the element that is being parsed and updated
   * @param tag the tagline in the xml of the element we are searching for
   */
  template<typename T>
  void parseElement(bool isReq, T *&ele, const std::string tag);


  /**
   * This function parses the current node of the tree and searches for all elements
   * with the tag of "tag". Once it is found, that element will be
   * parsed and the completed element will be returned.
   * this is a template function, so the type of the value that is taken in and
   * found are flexible. This will return all instances of a certain tag, this means it
   * can have 0..* elements.
   * If this variable is required and not found, an exception will be thrown.
   * @param isReq if this variable is required or not
   * @param eles the vector to be filled with found elements
   * @param tag the tagline in the xml of the elements we are searching for
   */
  template<typename T>
  void parseMultiElements(bool isReq, std::vector<T *> &eles, const std::string tag);


  /**
   * Recursive
   * This function takes in an element and a list of polymorphs. This function
   * will check the current ptree for all of the provided polymorphs. Once one
   * of the polymorphs is found, an element of that type is created over ele and
   * it's information is parsed out of the tree.
   * This function is a variadic template, which means it can take in any number
   * of objects each with their own type. However, this code is expecting each
   * type to be a variation of a polymorph and as so, this calls functions and
   * uses variables specific to the polymorph structure.
   * If this variable is required and not found, an exception will be thrown.
   * @param isReq if this variable is required or not
   * @param ele the element that will be updated
   * @param poly the current polymorph we are looking for in the xml
   * @param ARGS subsequent polymorphs to check in case the current is not found
   */
  template<typename T, typename X, typename... ARGS>
  void parsePolymorph(bool isReq, T *&ele, X poly, ARGS... args);

  /**
   * A base case for the recursive version of this function
   */
  template<typename T, typename X>
  void parsePolymorph(bool isReq, T *&ele, X poly);

  /**
   * Recursive
   * This function takes in a vector of elements and a list of polymorphs. This function
   * will check the current ptree for all of the provided polymorphs. All instances
   * of the polymorphs in the current tree are created through parsing and then added
   * to the vector.
   * This function is a variadic template, which means it can take in any number
   * of objects each with their own type. However, this code is expecting each
   * type to be a variation of a polymorph and as so, this calls functions and
   * uses variables specific to the polymorph structure.
   * If this variable is required and not found, an exception will be thrown.
   * @param isReq if this variable is required or not
   * @param eles a list of elements that will be updated
   * @param poly the current polymorph we are looking for in the xml
   * @param ARGS subsequent polymorphs to check in case the current is not found
   */
  template<typename T, typename X, typename... ARGS>
  void parseMultiPolymorphs(bool isReq, std::vector<T *> &eles, X poly, ARGS... args);

  /**
   * A base case for the recursive version of this function
   */
  template<typename T, typename X>
  void parseMultiPolymorphs(bool isReq, std::vector<T *> &eles, X poly);

  /**
   * This function will parse out a list of values that do not have tags associated
   * with the values. Keep in mind, if this function is being called, the values
   * can not be optional, this should only be used to parse a specific data structure.
   * This is a template function, whatever type is put in will be parsed out.
   * note: the type must have an overload for >> with input streams.
   */
  template<typename T>
  void parseTaglessValues(std::vector<T> &eles);

  /**
   * Virtual function
   * This is where we select what values we want to parse out of
   * the xml file by calling the parseX functions above.
   */
  virtual void parseValues() = 0;

  /**
   * This function read and parse the XML with root 'root'
   * as the base to parse the ptree
   * @param fileName xml file
   * @param root base level of xml
   */
  void parseXML(const std::string fileName, const std::string root);
};

template<typename T>
inline void ParseInterface::parseTaglessValues(std::vector<T> &eles)
{
  std::istringstream buf;
  buf.str(tree.get_value<std::string>());
  T temp;
  while (buf >> temp)
    eles.push_back(temp);
  buf.clear();
}

template<typename T>
inline void ParseInterface::parseElement(bool isReq, T *&ele, const std::string tag)
{
  auto child = tree.get_child_optional(tag);
  if (child) {
    ele = new T();
    ele->setTree(*child);
    ele->setParents(treeParents + "::" + tag);
    ele->parseValues();
  } else {
    if (isReq)
      throw(ParseException("Could not find " + treeParents + "::" + tag));
  }
}

template<typename T>
inline void ParseInterface::parsePrimitive(bool isReq, T &val, const std::string tag)
{
  boost::optional<T> newVal = tree.get_optional<T>(tag);
  if (newVal)
    val = *newVal;
  else if (isReq)
    throw(ParseException("Could not find " + treeParents + "::" + tag));
}

template<typename T>
inline void ParseInterface::parseMultiPrimitives(bool isReq, std::vector<T> &vals, const std::string tag)
{
  pt::ptree::const_iterator end = tree.end();
  for (pt::ptree::const_iterator it = tree.begin(); it != end; ++it) {
    if (it->first == tag) {
      T newVal;
      newVal = (it->second).get_value<T>();
      vals.push_back(newVal);
    }
  }
  if (isReq && vals.size() == 0)
    throw(ParseException("Could not find " + treeParents + "::" + tag));
}

template<typename T>
inline void ParseInterface::parseMultiElements(bool isReq, std::vector<T *> &eles, const std::string tag)
{
  pt::ptree::const_iterator end = tree.end();
  for (pt::ptree::const_iterator it = tree.begin(); it != end; ++it) {
    if (it->first == tag) {
      T *newEle = new T();
      newEle->setTree(it->second);
      newEle->setParents(treeParents + "::" + tag);
      newEle->parseValues();
      eles.push_back(newEle);
    }
  }
  if (isReq && eles.size() == 0)
    throw(ParseException("Could not find " + treeParents + "::" + tag));
}

template<typename T, typename X, typename... ARGS>
inline void ParseInterface::parsePolymorph(bool isReq, T *&ele, X poly, ARGS... args)
{
  auto child = tree.get_child_optional(poly.tag);
  if (child) {
    poly.setNewType(ele);
    ele->setTree(*child);
    ele->setParents(treeParents + "::" + poly.tag);
    ele->parseValues();
  } else
    parsePolymorph(isReq, ele, args...);
}


template<typename T, typename X>
inline void ParseInterface::parsePolymorph(bool isReq, T *&ele, X poly)
{
  auto child = tree.get_child_optional(poly.tag);
  if (child) {
    poly.setNewType(ele);
    ele->setTree(*child);
    ele->setParents(treeParents + "::" + poly.tag);
    ele->parseValues();
  } else if (isReq)
    throw(ParseException("Could not find " + treeParents + "::" + poly.tag));
}


template<typename T, typename X, typename... ARGS>
inline void ParseInterface::parseMultiPolymorphs(bool isReq, std::vector<T *> &eles, X poly, ARGS... args)
{
  pt::ptree::const_iterator end = tree.end();
  for (pt::ptree::const_iterator it = tree.begin(); it != end; ++it) {
    if (it->first == poly.tag) {
      T *newEle;
      poly.setNewType(newEle);
      newEle->setTree(it->second);
      newEle->setParents(treeParents + "::" + poly.tag);
      newEle->parseValues();
      eles.push_back(newEle);
    }
  }
  parseMultiPolymorphs(isReq, eles, args...);
}


template<typename T, typename X>
inline void ParseInterface::parseMultiPolymorphs(bool isReq, std::vector<T *> &eles, X poly)
{
  pt::ptree::const_iterator end = tree.end();
  for (pt::ptree::const_iterator it = tree.begin(); it != end; ++it) {
    if (it->first == poly.tag) {
      T *newEle;
      poly.setNewType(newEle);
      newEle->setTree(it->second);
      newEle->setParents(treeParents + "::" + poly.tag);
      newEle->parseValues();
      eles.push_back(newEle);
    }
  }
  if (isReq && eles.size() == 0)
    throw(ParseException("Could not find " + treeParents + "::" + poly.tag));
}

inline void ParseInterface::parseXML(const std::string fileName, const std::string root)
{
  pt::ptree t;

  try {
    pt::read_xml(fileName, t);
  } catch (boost::property_tree::xml_parser::xml_parser_error &e) {
    std::cerr << "Error reading tree in " << fileName << "\n";
    exit(EXIT_FAILURE);
  }


  auto child = t.get_child_optional(root);
  if (child) {
    setTree(*child);
    setParents("root::" + root);
    parseValues();
  } else {
    throw(ParseException("Could not find root::" + root));
  }
}
