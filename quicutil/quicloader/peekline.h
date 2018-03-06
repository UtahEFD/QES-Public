#ifndef PEEKLINE
#define PEEKLINE

#include <iostream>
#include <string>

inline void chomp(std::string& line)
{
	// chomp
	while
	(
	  line.size() > 0 && 
	  (
	    line.at(line.size() - 1) == '\n' || 
	    line.at(line.size() - 1) == '\r'
	  )
	) 
	{
		line = line.substr(0, line.size() - 1);
	}
}

inline std::istream& peekline(std::istream& input, std::string& line)
{
	line.clear();	
	int chars = 0;
	char c = 0;
	do
	{
		line += (c = input.get());
		chars++;
	}
	while(c != '\n' && !input.fail());
	
	chomp(line);
	
	for(; chars; chars--) {input.unget();}
	return input;
}

#endif

