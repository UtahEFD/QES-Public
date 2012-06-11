#ifndef ATTRIBUTELIST
#define ATTRIBUTELIST

#include "attribute.h"
#include "datamList.h"

class attributeList : public datamList
{

	public:
  
		attributeList();
		attributeList(attributeList const&);
		~attributeList();
		
		bool getValue(std::string const& _name, std::string& value_) const;
		bool setValue(std::string const& _name, std::string const& _value);

		void add(attribute const&);
		
		/**
		* Assume that only the set of attributes are in 
		* the istream. Do not provide a istream reference that has 
		* access to more than just the attributes!! Parse streams for
		* registered attributes until there are no more to parse. 
		*/
		friend std::istream& operator>>(std::istream&, attributeList&);

	protected:
	
		virtual std::istream& operator<<(std::istream&) = 0;
	
};

std::istream& operator>>(std::istream&, attributeList&);

#endif
