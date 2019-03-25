#ifndef __LANGUAGE_MAPPING__
#define __LANGUAGE_MAPPING__ 1

#include <iostream>
#include <boost/any.hpp>
#include<map>
#include <sstream>
#include <boost/lexical_cast.hpp>

// //////////////////////////////////////////////////////////////////
// 
// Class for holding the LANGUAGE MAPPING EVALUATION AND LANGUAGE MAP
// 
// //////////////////////////////////////////////////////////////////


class languageMap
{

public:

    languageMap() {}
    virtual ~languageMap(){}    
    virtual std::string retrieve(std::string variable_name); 
    virtual void modify_value(std::string variable_name,std::string newvalue);
    virtual void build_map(){

	std::cout<<"**************************************************************************************************************"<<std::endl;
	std::cout<<"---------------------------------------WARNING----------------------------------------------------------------"<<std::endl;
	std::cout<<"build_map of this class has not been overriden "<<std::endl;
	
	std::cout<<"**************************************************************************************************************"<<std::endl;


///steps:
///all predefined data types address can be pushed into the map
////// every class build_map should be called
//// for every class within it  make a void pointer and push the pointer
//// for every array  get the size in and each and every member location at zero

//  var_addressMap["buildings[]"]=sizeof(buildingData);         //size of the structure 
	////  var_addressMap["buildings.bldNum"]=&buildings[0].bldNum;
	};						///initally designed to omit changes in the constructor and have the programmer to override this method to build the map




protected:
std::map<std::string,boost::any>  var_addressMap;   								//change's made end

private:
 void* return_type(boost::any,std::string&);
};


#endif
