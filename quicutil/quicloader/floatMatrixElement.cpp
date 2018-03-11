#include "floatMatrixElement.h"
		
floatMatrixElement* floatMatrixElement::clone() const
{
	floatMatrixElement* fme = new floatMatrixElement(name);
	*fme->attributes = *attributes;
	fme->m = m; // copy the vector of values.
	return fme;
}
		
dim floatMatrixElement::getDimensions() const
{
	dim dimensions;
	dimensions.x = 0;
	dimensions.y = 0;
	dimensions.z = 0;
	
	std::stringstream ss;
	std::string dm;
	int i;
	
	if(attributes->getValue("xdim", dm))
	{
		ss.str(dm);
		ss >> i;
		dimensions.x = i;
		ss.clear();
	}
	else
	{
		dimensions.x = 0;
	}
	if(attributes->getValue("ydim", dm))
	{
		ss.str(dm);
		ss >> i;
		dimensions.y = i;
		ss.clear();
	}
	else
	{
		dimensions.y = 0;
	}
	if(attributes->getValue("zdim", dm))
	{
		ss.str(dm);
		ss >> i;
		dimensions.z = i;
		ss.clear();
	}
	else
	{
		dimensions.z = 0;
	}
	
	return dimensions;
}

void floatMatrixElement::registerAttributes()
{
	attributes->add(standardAttribute("xdim", "0"));
	attributes->add(standardAttribute("ydim", "0"));
	attributes->add(standardAttribute("zdim", "0"));
}
			
std::ostream& floatMatrixElement::operator>>(std::ostream& output) const
{
	std::cout << "# matrix" << std::endl;
	
	output << name << "[" << *attributes << "]" << std::endl;
	dim dimensions = this->getDimensions();
	
	for(int k = 0; k < dimensions.z; k++)
	{
		for(int j = 0; j < dimensions.y; j++)
		{
			for(int i = 0; i < dimensions.x; i++)
			{
				int ndx = k * dimensions.y * dimensions.x + j * dimensions.x + i;
				output << m[ndx] << " ";
			}
			output << std::endl;
		}
		output << std::endl << std::endl;
	}
	return output;
}

std::istream& floatMatrixElement::operator<<(std::istream& input)
{
	parseName(input);
	parseAttributes(input);
	
	dim dimensions = this->getDimensions();
	int mtrx_sz = dimensions.x * dimensions.y * dimensions.z;
	
	m.resize(mtrx_sz);
	
	for(int ndx = 0; ndx < mtrx_sz && !input.fail(); ndx++)
	{
		input >> m[ndx];
	}

	return input;
}

