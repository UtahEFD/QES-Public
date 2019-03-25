#include "fileParser.h"

		
fileParser::fileParser() : datamList() 
{
	filename = std::string("input.txt");
}

fileParser::fileParser(char const* _filename) : datamList() 
{
	filename = std::string(_filename);
}

fileParser::fileParser(std::string const& _filename) : datamList() 
{
	filename = _filename;
}

fileParser::fileParser(fileParser& _ifp) : datamList(_ifp)
{
	filename = _ifp.filename;
}

fileParser::~fileParser() {this->amnesia();}

void fileParser::commit(element& _element)
{
	datamList::push(&_element);
}

bool fileParser::recall(std::string const& _name, element** element_) const
{
	*element_ = (element*) datamList::search(_name);
	return (*element_ != NULL) ? (*element_)->foundQ() : false ;
}

bool fileParser::recall(std::string const& _name) const
{
	element* e = (element*) datamList::search(_name); 
	return (e != NULL) ? e->foundQ() : false ;
}

bool fileParser::recall(element const& _element) const
{
	return this->recall(_element.getName());	
}
		
void fileParser::forget(std::string const& name)
{
	datamList::remove(name);
}

void fileParser::amnesia() {datamList::clear();}
	
void fileParser::study()
{
	std::ifstream file(filename.c_str(), std::ifstream::in);
	if(!file.is_open()) 
	{
		std::cerr << "Unable to open " + filename + "." << std::endl;
		return;
	}
	
	remember(file);
		
	file.close();
}

void fileParser::study(std::string const& _filename)
{
	filename = _filename;
	this->study();
}

void fileParser::setFile(std::string const& _filename) {filename = _filename;}
std::string fileParser::getFile() const {return filename;}

