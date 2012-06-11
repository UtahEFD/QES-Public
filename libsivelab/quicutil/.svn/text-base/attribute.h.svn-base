#ifndef ATTRIBUTE
#define ATTRIBUTE

#include "datam.h"

class attribute : public datam
{
	public:
		
		attribute() : datam() {}
		attribute(std::string const& _name) : datam(_name) {}
		attribute(std::string const& _name, std::string const& _value) : datam(_name)
		{
			value = value;
			found = true;
		}
		
		std::string getValue() const;
		void setValue(std::string const&);

		attribute& operator=(attribute const& _att);
				
	protected:
	
		std::string value;

};

#endif
