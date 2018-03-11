#include "attributeList.h"

attributeList::attributeList() : datamList() {}

attributeList::attributeList(attributeList const& _attList) : datamList(_attList) 
{
	*this = _attList; // Use datam's = operator.
}

attributeList::~attributeList()
{
	// is the datamList destructor called?
}

bool attributeList::getValue(std::string const& _name, std::string& value_) const
{
	attribute* att = (attribute*) this->search(_name);
	if(att != NULL && att->foundQ())
	{
		value_ = att->getValue();
		return true;
	}
	else
	{
		return false;
	}
}

bool attributeList::setValue(std::string const& _name, std::string const& _value)
{
	attribute* att = (attribute*) datamList::search(_name);
	if(att != NULL)
	{
		att->setValue(_value);
		att->setFound(true);
		return true;
	}
	else
	{
		return false;
	}
}

void attributeList::add(attribute const& att)
{
	datamList::push(att.clone());
}
  
std::istream& operator>>(std::istream& input, attributeList& attList)
{	
	attList << input;
	return input;
}

