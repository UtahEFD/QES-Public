#ifndef FLOATVECTORELEMENT
#define FLOATVECTORELEMENT

#include <vector>
#include "standardElements.h"

class floatVectorElement : public standardElement
{
	public:
	
		floatVectorElement() : standardElement() 
		{
			// Register the attributes
			this->registerAttributes();
		}
		
		floatVectorElement(std::string const& _name) : standardElement(_name) 
		{
			// Register the attributes
			this->registerAttributes();
		}
		
		floatVectorElement* clone() const;
		// bool getDataAsLinearArray(float*) const;

	protected:

		std::vector<float> v;
		
		int getSize() const;
		void registerAttributes();
		
		std::ostream& operator>>(std::ostream&) const;
		std::istream& operator<<(std::istream&);
};

#endif

