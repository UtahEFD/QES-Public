#ifndef ELEMENT
#define ELEMENT

#include <string>
#include <iostream>

#include "attributeList.h"
#include "datam.h"

class element : public datam
{
	public:
		element() : datam() {attributes = NULL;}
		element(std::string const& _name) : datam(_name) {attributes = NULL;}
	  virtual ~element() {delete attributes;}
  
	  bool hasAttributeQ(std::string const&) const;
  
	  virtual void addAttribute(std::string const&) = 0;
	  virtual void addAttribute(std::string const&, std::string const&) = 0;
  
 		bool getAttributeValue(std::string const& _name, std::string& _value) const;
  	bool setAttributeValue(std::string const& _name, std::string const& _value);
  
	 protected:
		
		attributeList* attributes;
		  
	 private:  

};

#endif

