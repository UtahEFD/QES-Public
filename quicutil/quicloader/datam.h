#ifndef DATAM
#define DATAM

#include <string>
#include <iostream>

//#include "attributeList.h"

class datam
{
	public:

		datam();
		datam(std::string);

		virtual ~datam() {}

		std::string getName() const;
		void setName(std::string const&);
				
		bool foundQ() const;
		void setFound(bool);
		
		virtual datam* clone() const = 0;
		
		friend std::ostream& operator<<(std::ostream&, datam const&);
		friend std::istream& operator>>(std::istream&, datam&);
	
	protected:		

		std::string name;
		bool found;		

		virtual std::ostream& operator>>(std::ostream&) const = 0;		
		virtual std::istream& operator<<(std::istream&) = 0;
	
	private:
	
};

std::ostream& operator<<(std::ostream&, datam const&);
std::istream& operator>>(std::istream&, datam&);


#endif

