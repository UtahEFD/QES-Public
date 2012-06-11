#include "basicElements.h"		
	
floatElement* floatElement::clone() const
{
	floatElement* fe = new floatElement(datam::name);
	fe->value = value;
	fe->attributes = attributes;
	return fe;
}
		
std::ostream& floatElement::operator>>(std::ostream& output) const
{
	output << name << " " << std::flush;
	if(found) {output << value;} // \\todo what about the attributes?
	else			{output << "?";}
	//output << std::endl;
	return output; 
}

std::istream& floatElement::operator<<(std::istream& input)
{
	input >> name >> value;
	if(!input.fail()) {this->found = true;}
	return input;
}




	
intElement* intElement::clone() const
{
	intElement* ie = new intElement(datam::name);
	ie->value = value;
	ie->attributes = attributes;
	return ie;
}
		
std::ostream& intElement::operator>>(std::ostream& output) const
{
	output << name << " " << std::flush;
	if(found) {output << value;} // \\todo what about the attributes?
	else			{output << "?";}
	//output << std::endl;
	return output; 
}

std::istream& intElement::operator<<(std::istream& input)
{
	input >> name >> value;
	if(!input.fail()) {this->found = true;}
	return input;
}





boolElement* boolElement::clone() const
{
	boolElement* be = new boolElement(datam::name);
	be->value = value;
	be->attributes = attributes;
	return be;
}
		
std::ostream& boolElement::operator>>(std::ostream& output) const
{	
	output << name << " " << std::flush;
	if(found) {output << value;} // \\todo what about the attributes?
	else			{output << "?";}
	//output << std::endl;
	return output; 
}

std::istream& boolElement::operator<<(std::istream& input)
{
	input >> name >> value;
	if(!input.fail()) {this->found = true;}
	return input;
}





stringElement* stringElement::clone() const
{
	stringElement* se = new stringElement(name);
	se->value = value;
	se->attributes = attributes;
	return se;
}
		
std::ostream& stringElement::operator>>(std::ostream& output) const
{	
	output << name << " " << std::flush;
	if(found) {output << value;} // \\todo what about the attributes?
	else			{output << "?";}
	//output << std::endl;
	return output; 
}

std::istream& stringElement::operator<<(std::istream& input)
{
	input >> name >> value;
	if(!input.fail()) {this->found = true;}
	return input;
}

