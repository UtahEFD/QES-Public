/* File: Config.cpp
 * Author: Matthew Overby
 */

#include <cassert>
#include "Config.h"

using namespace SLUI;

Config::Config(OptionTracker *opt){

	m_optTracker = opt;
}

Config::~Config(){}

bool Config::checkFile(std::string filename){

	FILE *f;
	f = fopen(filename.c_str(), "rb");
	if(f == NULL) { return false; }
	return true;
}

void Config::updateConfig(){

	std::ofstream newConfig;
	newConfig.open("config");

	std::map<std::string, Keys*> keyList = m_optTracker->getKeys();
	for(std::map<std::string, Keys*>::iterator it = keyList.begin(); it != keyList.end(); it++){
		newConfig << "key " << it->first << "=" << it->second->toString() << "\n";
	}

	std::map<std::string, Option*> optionList = m_optTracker->getOptions();
	for(std::map<std::string, Option*>::iterator it = optionList.begin(); it != optionList.end(); it++){
	  if(it->second->type == OptionType::List){
	    newConfig << "list " << it->first << "=\"" << it->second->getString() << "\"\n";
	  }
	  else
		newConfig << "value " << it->first << "=" << it->second->getString() << "\n";
	}

	newConfig.close();
}

bool Config::loadConfig(){

	if(checkFile("config")){

		// Config file exists: parse
		std::ifstream oldConfig;
		oldConfig.open("config");
		
		if(oldConfig.is_open()) {
			
			std::string line;
			while (getline(oldConfig, line)) {
				
				std::string type;
				
				std::stringstream input(line);
				input >> type >> line;

				if(type.compare("key")==0){

					size_t pos = line.find("=");
					std::string command, value;

					if((int)pos == -1 && line.length() > 0) { 
						// equal sign not found in a parsed line
						std::cout << "**Error:  Bad Config at: " << line.c_str() << std::endl;
					}
					else{
						// else, extract command and value
						command = line.substr(0, pos);
						value = line.substr(pos+1);

						m_optTracker->bind(command, value);
					}
				} // end load keys
				else if(type.compare("value")==0){

					size_t pos = line.find("=");
					std::string command, value;

					if(pos == std::string::npos && line.length() > 0) { 
						// equal sign not found in a parsed line
						std::cout << "**Error:  Bad Config at: " << line.c_str() << std::endl;
					}
					else{
						// else, extract command and value
						command = line.substr(0, pos);
						value = line.substr(pos+1);

						m_optTracker->setString(command, value);
						m_optTracker->stateChanged(command);
					}

				} // end load values
				else if(type.compare("list")==0){

					size_t pos = line.find("=");
					size_t firstQuote = line.find_first_of("\"");
					size_t lastQuote = line.find_last_of("\"");
					std::string command, value;

					int numWords = 0;
					while(firstQuote == lastQuote){
					  numWords++;
					  std::string temp;
					  input >> temp;
					  line += " " + temp;
					  lastQuote = line.find_last_of("\"");
					  if(numWords > 100){
					    std::cout << "**Error:  Too many words in a list option!" << std::endl;
					    break;
					  }
					}

					if((pos == std::string::npos || firstQuote == std::string::npos
					    || lastQuote == std::string::npos) && line.length() > 0) { 
						// equal sign not found in a parsed line
						std::cout << "**Error:  Bad Config at: " << line.c_str() << std::endl;
						std::cout << firstQuote << " " << lastQuote << std::endl;
					}
					else{
						// else, extract command and value
						command = line.substr(0, pos);
						value = line.substr(firstQuote+1, lastQuote-firstQuote-1);

						m_optTracker->setString(command, value);
						m_optTracker->stateChanged(command);
					}
				}

			} // end of parsing config file
		} // end of opened config file

		oldConfig.close();
	} else {
		// The file was not found, create a new one
		updateConfig();
		return true;
	}

	return true;

}


