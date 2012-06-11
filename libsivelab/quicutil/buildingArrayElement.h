#ifndef BUILDINGARRAYELEMENT
#define BUILDINGARRAYELEMENT

#include <vector>
#include "standardElements.h"

class buildingElement : public standardElement
{
	public:	
		int id;
		float xfo;
		float yfo;
		float zfo;
		float hgt;
		float wth;
		float len;
		
		buildingElement() : standardElement() 
		{
			id = -1;
			xfo = yfo = zfo = hgt = wth = len = -1.;
			this->registerAttributes();
		}
		buildingElement(std::string const& _name) : standardElement(_name) 
		{
			id = -1;
			xfo = yfo = zfo = hgt = wth = len = -1.;
			this->registerAttributes();
		}
		virtual ~buildingElement() {}
		
		buildingElement* clone() const;
		
		std::ostream& operator>>(std::ostream&) const;
		std::istream& operator<<(std::istream&);
		
	protected:
		void registerAttributes();
		bool getValuesFromAttributes();

};


class buildingArrayElement : public standardElement
{
	public:	
		buildingArrayElement() : standardElement() {}
		buildingArrayElement(std::string const& _name) : standardElement(_name) {}
		virtual ~buildingArrayElement() {}
		
		buildingArrayElement* clone() const;
		
		int getNumBuildings() const;
		buildingElement getBuilding(unsigned int) const;
		
		std::ostream& operator>>(std::ostream&) const;
		std::istream& operator<<(std::istream&);
		
	protected:
		std::vector<buildingElement*> bldngs;			
};

#endif

