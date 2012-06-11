#ifndef LEGACYFILEPARSER
#define LEGACYFILEPARSER

#include "basicElements.h"
#include "fileParser.h"

class legacyFileParser : public fileParser
{
	public:
		
		/*
		* The first occurance of the name is used.
		* The format expected it 'name value'.
		* Lines starting with '#', '%', '!' and '//' are comments.
		* Lines containing an '!' are treated differently, values first, 
		* then label after '!'.
		*/
		void remember(std::istream& input);
	
	protected:
	
	private:

};

#endif

