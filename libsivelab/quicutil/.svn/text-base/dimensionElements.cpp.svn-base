#include "dimensionElements.h"
		
domainElement* domainElement::clone() const
{
	domainElement* de = new domainElement(name);
	*(de->attributes) = *attributes;
	return de;	
}

int domainElement::getX() const
{
	std::string size; 
	attributes->getValue("x", size);
	std::stringstream ss(size);
	
	int sz;
	return (!(ss >> sz).fail()) ? sz : -1 ;
}

int domainElement::getY() const
{
	std::string size; 
	attributes->getValue("y", size);
	std::stringstream ss(size);
	
	int sz;
	return (!(ss >> sz).fail()) ? sz : -1 ;
}

int domainElement::getZ() const
{
	std::string size; 
	attributes->getValue("z", size);
	std::stringstream ss(size);
	
	int sz;
	return (!(ss >> sz).fail()) ? sz : -1 ;
}

std::ostream& domainElement::operator>>(std::ostream& output) const
{
	output << name << "[" << *attributes << "]" << std::endl;
	return output;
}

std::istream& domainElement::operator<<(std::istream& input)
{
	parseName(input);
	parseAttributes(input);
	return input;
}

void domainElement::registerAttributes()
{
	attributes->add(standardAttribute("x", "0"));
	attributes->add(standardAttribute("y", "0"));
	attributes->add(standardAttribute("z", "0"));
}



resolutionElement* resolutionElement::clone() const
{
	resolutionElement* de = new resolutionElement(name);
	*de->attributes = *attributes;
	return de;
}

float resolutionElement::getDX() const
{
	std::string size; 
	attributes->getValue("dx", size);
	std::stringstream ss(size);
	
	float sz;
	return (!(ss >> sz).fail()) ? sz : 1. ;
}

float resolutionElement::getDY() const
{
	std::string size; 
	attributes->getValue("dy", size);
	std::stringstream ss(size);
	
	float sz;
	return (!(ss >> sz).fail()) ? sz : 1. ;
}

float resolutionElement::getDZ() const
{
	std::string size; 
	attributes->getValue("dz", size);
	std::stringstream ss(size);
	
	float sz;
	return (!(ss >> sz).fail()) ? sz : 1. ;
}

std::ostream& resolutionElement::operator>>(std::ostream& output) const
{
	output << name << "[" << *attributes << "]" << std::endl;
	return output;
}

std::istream& resolutionElement::operator<<(std::istream& input)
{
	parseName(input);
	parseAttributes(input);
	return input;
}

void resolutionElement::registerAttributes()
{
	attributes->add(standardAttribute("dx", "1"));
	attributes->add(standardAttribute("dy", "1"));
	attributes->add(standardAttribute("dz", "1"));
}
