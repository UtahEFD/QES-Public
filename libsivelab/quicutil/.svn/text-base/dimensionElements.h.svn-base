#ifndef DIMENSIONELEMENTS
#define DIMENSIONELEMENTS

#include "standardElements.h"

class domainElement : public standardElement
{
	public:	
		domainElement() : standardElement() 
		{
			this->registerAttributes();
		}
		domainElement(std::string const& _name) : standardElement(_name) 
		{
			this->registerAttributes();
		}
		
		domainElement* clone() const;
		
		int getX() const;
		int getY() const;
		int getZ() const;
		
		std::ostream& operator>>(std::ostream&) const;
		std::istream& operator<<(std::istream&);
		
	protected:
		void registerAttributes();				
};

class resolutionElement : public standardElement
{
	public:	
		resolutionElement() : standardElement() 
		{
			this->registerAttributes();
		}
		resolutionElement(std::string const& _name) : standardElement(_name)
		{
			this->registerAttributes();
		}
		
		resolutionElement* clone() const;
		
		float getDX() const;
		float getDY() const;
		float getDZ() const;
		
		std::ostream& operator>>(std::ostream&) const;
		std::istream& operator<<(std::istream&);
				
	protected:
		void registerAttributes();
};

#endif

