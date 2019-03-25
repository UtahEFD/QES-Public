#ifndef EMITTERARRAYELEMENT
#define EMITTERARRAYELEMENT

#include <vector>
#include "standardElements.h"

class emitterElement : public standardElement
{
	public:	
		int type;
		float sx;
		float sy;
		float sz;
		float ex;
		float ey;
		float ez;
		
		emitterElement() : standardElement() 
		{
			type = -1;
			sx = sy = sz = ex = ey = ez = -1.;
			this->registerAttributes();
		}
		emitterElement(std::string const& _name) : standardElement(_name) 
		{
			type = -1;
			sx = sy = sz = ex = ey = ez = -1.;
			this->registerAttributes();
		}
		virtual ~emitterElement() {}
		
		emitterElement* clone() const;
		
		std::ostream& operator>>(std::ostream&) const;
		std::istream& operator<<(std::istream&);
		
	protected:
		void registerAttributes();
		bool getValuesFromAttributes();

};


class emitterArrayElement : public standardElement
{
	public:	
		emitterArrayElement() : standardElement() {}
		emitterArrayElement(std::string const& _name) : standardElement(_name) {}
		virtual ~emitterArrayElement() {}
		
		emitterArrayElement* clone() const;
		
		int getNumEmitters() const;
		emitterElement getEmitter(unsigned int) const;
		
		std::ostream& operator>>(std::ostream&) const;
		std::istream& operator<<(std::istream&);
		
	protected:
		std::vector<emitterElement*> mttrs;
};

#endif

