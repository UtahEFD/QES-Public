#ifndef BASICELEMENTS
#define BASICELEMENTS

#include <string>
#include "standardElements.h"

class floatElement : public standardElement
{
	public:
		float value;
	
		floatElement() : standardElement() {value = 0.;}
		floatElement(std::string const& _name) : standardElement(_name) {value = 0.;}
		
		floatElement* clone() const;
		
		std::ostream& operator>>(std::ostream&) const;
		std::istream& operator<<(std::istream&);
		
	private:

};

class intElement : public standardElement
{
	public:
		int value;
	
		intElement() : standardElement() {value = 0;}
		intElement(std::string const& _name) : standardElement(_name) {value = 0;}
	
		intElement* clone() const;
		
		std::ostream& operator>>(std::ostream&) const;
		std::istream& operator<<(std::istream&);

	private:

};

class boolElement : public standardElement
{
	public:
		bool value;
	
		boolElement() : standardElement() {value = false;}
		boolElement(std::string const& _name) : standardElement(_name) {value = false;}

		boolElement* clone() const;

		std::ostream& operator>>(std::ostream&) const;
		std::istream& operator<<(std::istream&);

	private:

};

class stringElement : public standardElement
{
	public:
		std::string value;
	
		stringElement() : standardElement() {value = "";}
		stringElement(std::string const& _name) : standardElement(_name) {value = "";}

		stringElement* clone() const;

		std::ostream& operator>>(std::ostream&) const;
		std::istream& operator<<(std::istream&);
		
	private:

};

#endif
