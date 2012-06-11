#include "attribute.h"

std::string attribute::getValue() const {return value;}
void attribute::setValue(std::string const& _value) {value = _value;}

attribute& attribute::operator=(attribute const& _att)
{
	name  = _att.name;
	value = _att.value;
	found = _att.found;

	return *this;
}

