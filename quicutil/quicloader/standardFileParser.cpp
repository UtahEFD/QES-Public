#include "standardFileParser.h"

void standardFileParser::remember(std::istream& input)
{	
	std::string label;
	std::string line;
	
	do
	{
		peekline(input, line);
		
		//std::cout << line << std::endl;

		if(!line.empty()) 
		{
			if
			(
				line.find_first_of("#%;!") == 0 || 
				line.substr(0,2) == "//"
			)
			{
				//std::cout << "Comment found : " << line << std::endl;
				// Nothing to remember, just a comment.
				getline(input, line);
				continue;
			}
			else
			{
				std::stringstream ss(line, std::stringstream::in | std::stringstream::out);
				label.clear();
				ss >> label; // Could be crap...
				if(label.find("[") != std::string::npos)
				{
					label = label.substr(0, label.find("["));
				}
				
				//std::cout << "label = " << label << std::endl;	
				datam* cnddt = datamList::search(label);
					
				if(cnddt != NULL) 
				{
					input >> *cnddt; // Pull from input an appropriate entry.
					
					//std::cout << "name = " << cnddt->getName() << std::endl;
					//std::cout << "label = " << label << " search = " << cnddt->getName() << std::endl;
					//std::cout << "found = " << cnddt->foundQ() << std::endl;
				}
			}
		}		
		getline(input, line);
	}	
	while(!input.fail());
}


/*
void loveYourCommentIfYouLike(std::string const& line)
{
	if
	(
		line.find_first_of("#%;!") == 0 || 
		line.substr(0,2) == "//"
	)
	{
		// std::cout << "Comment found : " << line << std::endl;
		// Nothing to remember, just a comment.
		return;
	}
}
*/
	




