#include "floatVectorElement.h"

floatVectorElement* floatVectorElement::clone() const
{
	floatVectorElement* fve = new floatVectorElement(name);
	*fve->attributes = *attributes;
	fve->v = v; // copy the vector of values.
	return fve;
}

void floatVectorElement::registerAttributes()
{
	attributes->add(standardAttribute("size", "0"));
}

int floatVectorElement::getSize() const
{
	std::string size; 
	attributes->getValue("size", size);
	std::stringstream ss(size);
	
	int sz;
	return (!(ss >> sz).fail()) ? sz : -1 ;
}

std::ostream& floatVectorElement::operator>>(std::ostream& output) const
{
	output << name << " [" << *attributes << "] " << std::endl;
	
	for(unsigned int i = 0; i < v.size(); i++)
	{
		output << v[i] << " " << std::flush;
	}
	
	output << std::endl;
	
	return output;
}

std::istream& floatVectorElement::operator<<(std::istream& input)
{	
	if(parseName(input) && parseAttributes(input))
	{
		// All that's left should be the data.
		// Get the data.
		float tmp = 0.0f;
		int v_size = this->getSize();	
		for(int i = 0; i < v_size; i++)
		{
			input >> tmp;
			v.push_back(tmp);
		}
	}
	
	this->found = (!input.fail()) ? true : false ;
	
	return input;
}
		
