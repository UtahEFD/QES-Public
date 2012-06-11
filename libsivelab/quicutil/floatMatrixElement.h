#ifndef FLOATMATRIXELEMENT
#define FLOATMATRIXELEMENT

#include <vector>
#include "standardElements.h"

typedef struct dim
{
	int x;
	int y;
	int z;
} dim;

class floatMatrixElement : public standardElement
{
	public:
	
		floatMatrixElement() : standardElement() 
		{
			this->registerAttributes();
		}
		
		floatMatrixElement(std::string const& _name) : standardElement(_name) 
		{
			this->registerAttributes();			
		}
		
		floatMatrixElement* clone() const;
		
		dim getDimensions() const;

	protected:

		std::vector<float> m;
		
		void registerAttributes();
		void determineDimensions();
		
		std::ostream& operator>>(std::ostream&) const;
		std::istream& operator<<(std::istream&);

};

#endif
