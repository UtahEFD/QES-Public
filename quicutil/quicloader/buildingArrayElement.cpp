#include "buildingArrayElement.h"

buildingElement* buildingElement::clone() const
{
	buildingElement* de = new buildingElement(name);
	*(de->attributes) = *attributes;
	return de;
}

std::ostream& buildingElement::operator>>(std::ostream& output) const
{
	output << "building[id " << id;
	output << " xfo " << xfo;
	output << " yfo " << yfo;
	output << " zfo " << zfo;
	output << " x " << len;
	output << " y " << wth;
	output << " z " << hgt;
	output << "]" << std::endl;
	return output;
}

std::istream& buildingElement::operator<<(std::istream& input)
{
	parseName(input);
	parseAttributes(input);
	getValuesFromAttributes();
	return input;
}
		
void buildingElement::registerAttributes()
{
	attributes->add(standardAttribute("id", "0"));
	attributes->add(standardAttribute("xfo", "0"));
	attributes->add(standardAttribute("yfo", "0"));
	attributes->add(standardAttribute("zfo", "0"));
	attributes->add(standardAttribute("x", "0"));
	attributes->add(standardAttribute("y", "0"));
	attributes->add(standardAttribute("z", "0"));
}

bool buildingElement::getValuesFromAttributes()
{
	std::string att_val;
	std::stringstream ss(std::stringstream::in | std::stringstream::out);
	attributes->getValue("id",  att_val); ss.str(att_val); ss >> id;  ss.clear();
	attributes->getValue("xfo", att_val); ss.str(att_val); ss >> xfo; ss.clear();
	attributes->getValue("yfo", att_val); ss.str(att_val); ss >> yfo; ss.clear();
	attributes->getValue("zfo", att_val); ss.str(att_val); ss >> zfo; ss.clear();
	attributes->getValue("x",   att_val); ss.str(att_val); ss >> len; ss.clear();
	attributes->getValue("y",   att_val); ss.str(att_val); ss >> wth; ss.clear();
	attributes->getValue("z",   att_val); ss.str(att_val); ss >> hgt;	ss.clear();
	return true;
}






buildingArrayElement* buildingArrayElement::clone() const
{
	buildingArrayElement* de = new buildingArrayElement(name);
	de->bldngs = bldngs;
	return de;	
}

int buildingArrayElement::getNumBuildings() const
{
		// Note: size() returns a std::vector::size_type which cannot be negative, but we are returning an int here, which can be.  What to do?
	std::vector<buildingElement*>::size_type sz = bldngs.size();
	return static_cast<int>(sz);
}

buildingElement buildingArrayElement::getBuilding(unsigned int ndx) const
{
	if(ndx < 0 || bldngs.size() <= ndx) {return buildingElement();}
	
	return *bldngs[ndx];
}

std::ostream& buildingArrayElement::operator>>(std::ostream& output) const
{
	std::cout << "# building array" << std::endl;
	for(unsigned int i = 0; i < bldngs.size(); i++)
	{
		output << *bldngs[i] << std::flush;
	}
	return output;
}

std::istream& buildingArrayElement::operator<<(std::istream& input)
{
	buildingElement* b = new buildingElement(name);
	input >> *b;
	bldngs.push_back(b);
	return input;
}

