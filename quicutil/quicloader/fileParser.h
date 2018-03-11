#ifndef FILEPARSER
#define FILEPARSER

#include <iostream>
#include <fstream>
#include <sstream>

#include "datamList.h"
#include "element.h"
#include "peekline.h"

class fileParser : public datamList
{
	public:
	
		fileParser();
		fileParser(char const*);
		fileParser(std::string const&);
		fileParser(fileParser&);
		~fileParser();
		
		void commit(element&);
		
		bool recall(std::string const& _name, element** element_) const;
		bool recall(std::string const& _name) const;
		bool recall(element const&) const;
		void forget(std::string const& _name);
		void amnesia();
		
		/**
		* Search the given file from the start for each committed name.
		*/
		void study();
		void study(std::string const&);
		
		void setFile(std::string const&);
		std::string getFile() const;
		
	protected:
	
		std::string filename;

		/*
		* This method must be implemented in order to determine how you want
		* your app specific fileparser to parse the file. 
		*
		* use insertion operator to populate datam.
		*
		* @param input, an input stream for the file
		*/
		virtual void remember(std::istream&) = 0;		
	
	private:
};

#endif

