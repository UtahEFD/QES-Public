#include "standardElements.h"

standardAttribute* standardAttribute::clone() const
{
	standardAttribute* sAtt = new standardAttribute(name);
	sAtt->setValue(this->getValue());
	return sAtt;	
}
		
std::ostream& standardAttribute::operator>>(std::ostream& output) const
{
	output << name << " " << ((found) ? value : "?") << " " << std::flush;
	return output;
}

std::istream& standardAttribute::operator<<(std::istream& input)
{
	input >> name >> value;
	if(!input.fail()) {found = true;}
	return input;
}



std::istream& standardAttributes::operator<<(std::istream& input)
{
	std::string att;
	std::string val;
	
	while(input >> att >> val)
	{
		this->setValue(att, val);
	}
	
	return input;
}


void standardElement::addAttribute(std::string const& _name)
{
	attributes->add(standardAttribute(_name));
}

void standardElement::addAttribute(std::string const& _name, std::string const& _value)
{
	standardAttribute* sAtt = new standardAttribute(_name);
	sAtt->setValue(_value);
	attributes->add(*sAtt);
}

bool standardElement::attributesFoundQ(std::istream& input) const
{
	std::string line;
	peekline(input, line);
	if(!input.fail() && line.find('[') != std::string::npos)
	{
		return true;
	}
	else
	{
		return false;
	}	
}

bool standardElement::parseName(std::istream& input)
{
	std::stringbuf sbname;
	std::stringstream ssname;

/*
	if(!attributesFoundQ(input))
	{
		input >> name;
	}
*/

	if(attributes->isEmptyQ())
	{
		input >> name;
		if(name.find("[") != std::string::npos)
		{
			std::cerr << "Found a '[' for an element that doesn't have attributes." << std::endl;
			std::cerr << "Dropping attributes for " << name << "..." << std::flush;
			name = name.substr(0, name.find("["));
			
			std::stringbuf dump;
			input.get(dump, ']');
			input.get(); // Toss the right bracket as well.
			if(input.fail()) 
			{
				std::cerr << "Unable to find matching ']'." << std::endl;
				return false;
			}
			std::cerr << "Now we're hopefully back on track." << std::endl;
		}
	}
	else
	{
		input.get(sbname, '[');
		if(input.fail()) 
		{
			std::cerr << "Error parsing name, looking for '['." << std::endl;
			std::cerr << "The name should be followed by '['." << std::endl;
			return false;
		}
		// Leave the bracket for getting attributes.
		// input.get();	
		ssname.str(sbname.str());
		ssname >> name;
	}
	//std::cout << "Parsed the name : " << name << std::endl;
	
	return (!input.fail() && !ssname.fail()) ? true : false ;
}

bool standardElement::parseAttributes(std::istream& input)
{
	if(!attributesFoundQ(input)) 
	{
		std::cerr << "Didn't find attributes for " << name << "." << std::endl;
		return false;
	}
	if(attributes->isEmptyQ()) 
	{
		std::cerr << name << " has not attributes to be parsed." << std::endl;
		return false;
	}

	std::stringbuf sbatts;

	// Get the attributes and kill the bracket.	
	char l_brckt;
	char r_brckt;
	
	input >> l_brckt;
	if(l_brckt != '[')
	{
		std::cerr << "Error parsing complex element attributes while looking for '['." << std::endl;
		return false;
	}
	input.get(sbatts, ']');
	if(input.fail())
	{
		std::cerr << "Error parsing complex element attributes." << std::endl;
		return false;
	}
	input >> r_brckt;
	if(r_brckt != ']')
	{
		std::cerr << "Error parsing complex element attributes, while looking for ']'." << std::endl;
		return false;
	}
	
	std::stringstream ssatts(sbatts.str());
	ssatts >> *attributes; // Rocks ssatts to failure.

	//std::cout << "Parsed the attributes : " << *attributes << std::endl;
	
	return (!input.fail()) ? true : false ;
}


