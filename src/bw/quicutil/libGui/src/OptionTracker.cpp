/* File: OptionTracker.cpp
 * Author: Matthew Overby
 */

#include "OptionTracker.h"

/** TODO
* Since many of these functions (getValue, getActive, etc...) are 
* called so frequently (many times per frame), they need to be optimized.
*/

using namespace SLUI;

OptionTracker::OptionTracker(){
}

OptionTracker::~OptionTracker(){

	std::map< std::string, Option* >::iterator it1;
	for ( it1 = options.begin(); it1 != options.end(); it1++ ) {

		Option *delPtr = it1->second;
		if( delPtr ){
			delete delPtr;
			options.erase( it1 );
		}
	}

	std::map< std::string, Keys* >::iterator it2;
	for ( it2 = keys.begin(); it2 != keys.end(); it2++ ) {

		Keys *delPtr = it2->second;
		if( delPtr ){
			delete delPtr;
			keys.erase( it2 );
		}
	}
}

void OptionTracker::addOption( std::string command, Option *newOption ){

	std::map<std::string, Option*>::iterator itr = options.find( command );
	if(itr == options.end()){
		options[command] = newOption;
	}
}

void OptionTracker::addBoolOption(std::string command, bool init){

	// Create an option to insert and insert it
	options[command] = new BoolOption( init );

	// Check to make sure the option was created properly. If we didn't, show error.
	std::map<std::string, Option*>::iterator c_itr = options.find(command);
	if(c_itr == options.end())
		std::cout << "ERROR: Creating bool option " << command << std::endl;
}

void OptionTracker::toggle(std::string command){

	std::map<std::string, Option*>::iterator option_itr = options.find(command);

	if(option_itr != options.end()){
		if(option_itr->second->type == OptionType::Bool){
			option_itr->second->toggle();
		}
		else{
			std::cout << "ERROR: "<< command << " is not a bool option, cannot toggle" << std::endl;
		}
	}
}

bool OptionTracker::getActive(std::string command){

	std::map<std::string, Option*>::iterator option_itr = options.find(command);

	if(option_itr != options.end()){
		if(option_itr->second->type == OptionType::Bool){
			return option_itr->second->getValue();
		}
	}

	return false;
}

void OptionTracker::setActive(std::string command, bool newActive){

	std::map<std::string, Option*>::iterator option_itr = options.find(command);

	if(option_itr != options.end()){
		if(option_itr->second->type == OptionType::Bool){
			option_itr->second->setValue(newActive);
		}
	}
}

void OptionTracker::addValueOption(std::string command, float init){

	// Create an option to insert and insert it
	options[command] = new ValueOption(init);

	// Check to make sure the option was created properly. If we didn't, show error.
	std::map<std::string, Option*>::iterator c_itr = options.find(command);
	if(c_itr == options.end())
		std::cout << "ERROR: Creating bool option " << command << std::endl;
}

float OptionTracker::getValue(std::string command){

	std::map<std::string, Option*>::iterator option_itr = options.find(command);

	if(option_itr != options.end()){
		if(option_itr->second->type == OptionType::Value ||
			option_itr->second->type == OptionType::Bool ){
			return option_itr->second->getValue();
		}
	}

	return 0;
}

void OptionTracker::setValue(std::string command, float newValue){

	std::map<std::string, Option*>::iterator option_itr = options.find(command);

	if(option_itr != options.end()){
		if(option_itr->second->type == OptionType::Value ||
			option_itr->second->type == OptionType::Bool ){
			option_itr->second->setValue(newValue);
		}
	}
}

void OptionTracker::setMinMax(std::string command, float min, float max){

	std::map<std::string, Option*>::iterator option_itr = options.find( command );

	if(option_itr != options.end()){
		if(option_itr->second->type == OptionType::Value){
			option_itr->second->setMinMax(min, max);
		}
	}
}
 
std::string OptionTracker::getString(std::string command){

	std::map<std::string, Option*>::iterator option_itr = options.find(command);

	if(option_itr != options.end()){
		return option_itr->second->getString();
	}
	else{
		std::map<std::string, Keys*>::iterator keys_itr = keys.find(command);

		if(keys_itr != keys.end()){
			return keys_itr->second->toString();
		}
	}

	return "";
}

void OptionTracker::setString(std::string command, std::string newValue){

	std::map<std::string, Option*>::iterator option_itr = options.find(command);

	if(option_itr != options.end()){
		option_itr->second->setString(newValue);
	}
}

bool OptionTracker::stateChanged(std::string command){

	std::map<std::string, Option*>::iterator option_itr = options.find(command);

	if(option_itr != options.end()){
		return option_itr->second->stateChanged();
	}

	return false;
}

void OptionTracker::addListOption(std::string command, std::vector< std::string > opts){

	// Create an option to insert and insert it
	options[command] = new ListOption(opts);

	// Check to make sure the option was created properly. If we didn't, show error.
	std::map<std::string, Option*>::iterator c_itr = options.find(command);
	if(c_itr == options.end())
		std::cout << "ERROR: Creating list option " << command << std::endl;
}

std::string OptionTracker::getListValue(std::string command){

	std::map<std::string, Option*>::iterator option_itr = options.find(command);

	if(option_itr != options.end()){
		if(option_itr->second->type == OptionType::List){
			return option_itr->second->getString();
		}
	}
	return "";
}

void OptionTracker::setListValue(std::string command, std::string value){

	std::map<std::string, Option*>::iterator option_itr = options.find(command);

	if(option_itr != options.end()){
		if(option_itr->second->type == OptionType::List){
			option_itr->second->setString(value);
		}
	}
}

void OptionTracker::addKeyOption(std::string action, std::string init){

	// Create a key to insert and insert it
	keys[action] = new Keys(init);

	// Check to make sure the key was created properly. If we didn't, show error.
	std::map<std::string, Keys*>::iterator c_itr = keys.find(action);
	if(c_itr == keys.end())
		std::cout << "ERROR: Creating key option " << action << std::endl;
}

sf::Key::Code OptionTracker::getKey(std::string action){

	std::map<std::string, Keys*>::iterator keys_itr = keys.find(action);

	if(keys_itr != keys.end()){
		return keys_itr->second->getKeyCode();
	}

	return sf::Key::Escape;
}

std::string OptionTracker::getKeyStr(std::string action){

	std::map<std::string, Keys*>::iterator keys_itr = keys.find(action);

	if(keys_itr != keys.end()){
		return keys_itr->second->toString();
	}

	return " ";
}

bool OptionTracker::bind(std::string action, std::string desiredKey){

	if(desiredKey.length() > 0 && action.length() > 0){

		std::map<std::string, Keys*>::iterator keys_itr = keys.find(action);

		if(keys_itr != keys.end()){
			return keys_itr->second->setNewKey(desiredKey);
		}
		else return false;
	}
	else return false;
}

std::map<std::string, Option*> OptionTracker::getOptions(){
	return options;
}

std::map<std::string, Keys*> OptionTracker::getKeys(){
	return keys;
}


