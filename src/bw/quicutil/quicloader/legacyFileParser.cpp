#include "legacyFileParser.h"

void legacyFileParser::remember(std::istream& input)
{	
	std::string label;
	std::string value;
	
	std::string line;
	std::stringstream ss(std::stringstream::in | std::stringstream::out);
	
	//std::cout << "Remembering..." << std::endl;
	
	do
	{
		getline(input, line);
		chomp(line);
		ss.str(line);
	
		//std::cout << line << std::endl;
	
		if
		(
			line.find_first_of("#%;!") == 0 || 
			line.substr(0,2) == "//"
		)
		{
			//std::cout << "Comment found : " << line << std::endl;
			// Nothing to remember, just a comment.
			continue;
		}
		else
		{
			if(line.find_first_of("!") != std::string::npos)
			{
				// Then the value should be the first thing encountered.
				ss >> value;
				// Kill the "!".
				label = line.substr(line.find_first_of("!") + 1);
				//ss >> label;
				//std::cout << "label = " << label << std::endl;
				//if(label == "!")
				//{
				//	ss >> label; // Grab the next "word".
				//}
				//else
				//{
				//	label = label.substr(label.find_first_of("!") + 1); // Kill the "!".
				//}
			}
			else
			{
				ss >> label >> value;
			}
		}

		// Part of the hack to get into standardElements (only).
		ss.str(value + " " + value + " ");
	
		//std::cout << "ss = " << ss.str() << std::endl;
	
		// Push into datamList.
		datam* cnddt = datamList::search(label);
		if(cnddt != NULL) 
		{
			// Hack it together!!
			ss >> *cnddt;
			
			//std::cout << "cnndt = " << *cnddt << std::endl;
			cnddt->setName(label);
			
			//std::cout << *cnddt << std::endl;			
			//std::cout << "found = " << cnddt->foundQ() << std::endl;
		}
		
		ss.clear(); ss.str("");
	}	
	while(!input.fail());
}

