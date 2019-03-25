/* File: Option.cpp
 * Author: Matthew Overby
 */

#include "Option.h"

using namespace SLUI;


/*
*	Base Class
*/

void Option::setToDefault(){}

float Option::getValue() { return 0; }

void Option::toggle(){}

void Option::setString(std::string newValue){}

void Option::setValue(float newValue){}

std::string Option::getString(){ return ""; }

void Option::setMinMax(float min, float max){}

bool Option::stateChanged(){

	if(stateChangedFlag){
		stateChangedFlag = false;
		return true;
	}

	return false;
}


/*
*	Bool Option
*/

BoolOption::BoolOption( bool init ){

	defaultActive = init;
	active = init;
	type = OptionType::Bool;
	stateChangedFlag = false;
}

void BoolOption::toggle(){

	if(active){
		active = false;
		stateChangedFlag = true;
	}
	else{
		active = true;
		stateChangedFlag = true;
	}
}

void BoolOption::setString(std::string val){

	if(val.compare("off")==0 || val.compare("0")==0){
		active = false;
		stateChangedFlag = true;
	}
	else if(val.compare("on")==0 || val.compare("1")==0){
		active = true;
		stateChangedFlag = true;
	}
}

std::string BoolOption::getString(){

	std::string result;

	if(active){
		result = "on";
	}
	else{
		result = "off";
	}

	return result;
}

void BoolOption::setToDefault(){
	active = defaultActive;
	stateChangedFlag = true;
}

float BoolOption::getValue(){

	if( active ) return 1;
	return 0;
}

void BoolOption::setValue(float value){

	if( value == 0 ) active = false;
	else if( value == 1 ) active = true;
	stateChangedFlag = true;
}


/*
*	Bool Option
*/


ValueOption::ValueOption( float init ){

	type = OptionType::Value;
	value = init;
	stateChangedFlag = false;
	defaultValue = value;
	bounded = false;
	min = 0;
	max = 0;
}

void ValueOption::setToDefault(){

	bounded = false;
	value = defaultValue;
	stateChangedFlag = true;
}

void ValueOption::setMinMax( float newMin, float newMax ){

	bounded = true;
	min = newMin;
	max = newMax;
	if( value > newMax ) value = newMax;
	if( value < newMin ) value = newMin;
}

void ValueOption::setString(std::string val){

	float newVal;
	if( sscanf( val.c_str(), "%f", &newVal ) != EOF ){
		setValue( newVal );
	}

}

std::string ValueOption::getString(){

	std::stringstream result;
	result << value;
	return result.str();
}

float ValueOption::getValue(){

	return value;
}

void ValueOption::setValue(float newVal){

	if( bounded ){
		if( newVal > max ) value = max;
		else if( newVal < min ) value = min;
		else value = newVal;
	}
	else{
		value = newVal;
	}

	stateChangedFlag = true;
}

/*
*	List Option
*/

ListOption::ListOption( std::vector<std::string> init ){

	type = OptionType::List;
	listOptions = init;
	stateChangedFlag = false;
	index = 0;
}

void ListOption::setString(std::string newValue){

	for(int i=0; i<listOptions.size(); i++){
		if(listOptions.at(i).compare( newValue ) == 0 && index != i){
			index = i;
			stateChangedFlag = true;
		}
	}
}

std::string ListOption::getString(){

	std::stringstream result;
	result << listOptions[index];
	return result.str();
}

/*
*	Time Option
*/

TimeOption::TimeOption( std::string init ){

	std::stringstream time( init );

	std::string val1 = "";
	std::string val2 = "";
	std::string val3 = "";

	getline(time, val1, ':');
	getline(time, val2, ':');
	time >> val3;

	// I know atoi is bad but since 0 is the time default
	// it will work just fine
	hour = atoi(val1.c_str());
	minute = atoi(val2.c_str());
	second = atoi(val3.c_str());

	if( hour > 23 ) hour = 23;
	if( hour < 0 ) hour = 0;
	if( minute > 59 ) minute = 59;
	if( minute < 0 ) minute = 0;
	if( second > 59 ) second = 59;
	if( second < 0 ) second = 0;

	default_hour = hour;
	default_minute = minute;
	default_second = second;
}

void TimeOption::setToDefault(){

	hour = default_hour;
	minute = default_minute;
	second = default_second;

	stateChangedFlag = true;
}

void TimeOption::setString( std::string newValue ){

	std::stringstream time( newValue );

	std::string val1 = "";
	std::string val2 = "";
	std::string val3 = "";

	getline(time, val1, ':');
	getline(time, val2, ':');
	time >> val3;

	// I know atoi is bad but since 0 is the time default
	// it will work just fine
	hour = atoi(val1.c_str());
	minute = atoi(val2.c_str());
	second = atoi(val3.c_str());

	if( hour > 23 ) hour = 23;
	if( hour < 0 ) hour = 0;
	if( minute > 59 ) minute = 59;
	if( minute < 0 ) minute = 0;
	if( second > 59 ) second = 59;
	if( second < 0 ) second = 0;

	stateChangedFlag = true;
}

std::string TimeOption::getString(){

	std::stringstream time("");
	time << hour << ":" << minute << ":" << second;

	return time.str();
}




