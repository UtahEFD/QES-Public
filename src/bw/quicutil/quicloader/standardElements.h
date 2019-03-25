#ifndef STANDARDELEMENTS
#define STANDARDELEMENTS

#include "element.h"
#include "attribute.h"
#include "peekline.h"

class standardAttribute : public attribute
{
	public:
		standardAttribute() : attribute() {}
		standardAttribute(std::string const& _name) : attribute(_name) {}
		standardAttribute(std::string const& _name, std::string const& _value) : attribute(_name, _value) {}
		virtual ~standardAttribute() {}
	
		standardAttribute* clone() const;
		
		std::ostream& operator>>(std::ostream&) const;
		std::istream& operator<<(std::istream&);
};

class standardAttributes : public attributeList
{
	public:
		standardAttributes() : attributeList() {}
		virtual ~standardAttributes() {}

	protected:
		
		// Assumes that the incoming istream only has attribute value pairs
		// and nothing more or less. Parses these in the standard way.
		std::istream& operator<<(std::istream&);
};

class standardElement : public element
{
	public:
	
		standardElement() : element() 
		{
			attributes = new standardAttributes();
		}
		standardElement(std::string const& _name) : element(_name) 
		{
			attributes = new standardAttributes();
		}
		virtual ~standardElement() {}
	
		void addAttribute(std::string const&);
		void addAttribute(std::string const&, std::string const&);
		
		bool attributesFoundQ(std::istream&) const;
		
		bool parseName(std::istream&);
		bool parseAttributes(std::istream&);
};

#endif

