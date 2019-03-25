#include "element.h"

bool element::hasAttributeQ(std::string const& _attrib) const
{
	if(attributes == NULL) {return false;}
  //search attribute list for attribute
  std::string dmmy;
  return attributes->getValue(_attrib, dmmy);
}

bool element::getAttributeValue(std::string const& _name, std::string& _value) const
{
	if(attributes == NULL) {return false;}
	return attributes->getValue(_name, _value);
}

bool element::setAttributeValue(std::string const& _name, std::string const& _value)
{
	if(attributes == NULL) {return false;}
  return attributes->setValue(_name, _value);
}

