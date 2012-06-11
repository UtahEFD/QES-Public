#ifndef FIRST_H
#define FIRST_H 1
#include <boost/any.hpp>
#include <iomanip>
#include<iostream>
#include<string>
using namespace std;
#include "base.h"
#include <sstream>
#include <vector>
//using this to simulate the base class of quic data

class first : public base
{
public:
first(){

 x_subdomain_sw=0;
  y_subdomain_sw=0;
  x_subdomain_ne=0;
  y_subdomain_ne=0;
  myname="rockstar";
  zo=0;
build_map();


}

 int x_subdomain_sw;
  int y_subdomain_sw;
  int x_subdomain_ne;
  int y_subdomain_ne;

  float zo;
  string myname;
  struct buildingData
  {
    string bldName;
    int bldNum;
    int group;
    int type;
    float height;
    float width;
    float length;
    float xfo;
    float yfo;
    float zfo;
    float gamma;
    float supplementalData;
  };

  std::vector<buildingData> buildings;


void build_map(){
 buildings.resize(5);
  
	///building the map
          my_map["myname"]=&myname;
	  my_map["x_subdomain_sw"]=&x_subdomain_sw;
	  my_map["y_subdomain_sw"]=&y_subdomain_sw;
	  my_map["x_subdomain_ne"]=&x_subdomain_ne;
	  my_map["y_subdomain_ne"]=&y_subdomain_ne;
	  my_map["zo"]=&zo;
	  my_map["buildings[]"]=sizeof(buildingData);     
	  cerr<<"-------------------------------------size of structure is "<<sizeof(buildingData);    //size of the structure 
	  my_map["buildings.bldNum"]=&buildings[0].bldNum;
	  my_map["buildings.group"]=&buildings[0].group;
	    
	  my_map["buildings.type"]=&buildings[0].type;
	  my_map["buildings.height"]=&buildings[0].height;
	    
	  my_map["buildings.width"]=&buildings[0].width;
	  my_map["buildings.length"]=&buildings[0].length;
	    
	  my_map["buildings.xfo"]=&buildings[0].xfo;
	  my_map["buildings.yfo"]=&buildings[0].yfo;
	    
	  my_map["buildings.zfo"]=&buildings[0].zfo;
	  my_map["buildings.gamma"]=&buildings[0].gamma;
	  my_map["buildings.supplementalData"]=&buildings[0].supplementalData;
	  //my_map["buildings.type_enum"]=&buildings[0].type_enum;

}
void print()
{

	if(buildings.size()<2)
		buildings.resize(5);
	  cout<<"\nthis is first class"<<endl;
	  cout<<"x_subdomain_sw:"<<x_subdomain_sw<<endl;
	  cout<<"y_subdomain_sw:"<<y_subdomain_sw<<endl;	
	   cout<<"x_subdomain_ne:"<<x_subdomain_ne<<endl;
	  cout<<"y_subdomain_ne:"<<y_subdomain_ne<<endl;
           cout<<"zo:"<<zo<<endl;
	   cout<<"myname:"<<myname<<endl;

	cout<<"gamma of buildings 0:\t"<<buildings[0].gamma<<endl;
        cout<<"gamma og buildigns 1:\t"<<buildings[1].gamma<<endl;
	cout<<"xfo of builiding 0:\t"<<buildings[0].xfo<<endl;
	cout<<"xfo of builiding 1:\t"<<buildings[1].xfo<<endl;
	 // cout<<"y_subdomain_sw:"<<y_subdomain_sw<<endl;	
}

 boost::any retrieve(std::string str)
{
      
	boost:: any r;
	if(!str.compare("value"))
	{

		r=value;
		return r;
	
	}
        else
	if(!str.compare("step"))
	{
		r=step;
		return r;
	
		
	}
	else
	if(!str.compare("name"))
	{
		r=name;
		return r;
	
		
	}
	//int temp;
	//temp=boost::any_cast<int>(r);
	//std::cout<<"tempo is"<<temp<<std::endl;
	
	

}
bool modify(std::string str,boost::any newvalue)
{
    
	if(!str.compare("value"))
	{
		value=boost::any_cast<int>(newvalue);
		return true;
	}
        else
	if(!str.compare("step"))
	{
	
		step=boost::any_cast<float>(newvalue);
		return true;
	
		
	}
	else
	if(!str.compare("name"))
	{
		name=boost::any_cast<string>(newvalue);
		return true;
	
		
	}


}
 bool modify_v2(std::string str,std::string newvalue)
{
	stringstream temp;
	
	if(!str.compare("x_subdomain_sw"))
	{
		
		temp<<newvalue;
		temp>>x_subdomain_sw;
              //  cout<<"value"<<value<<endl;
		return true;
	}
        else
	if(!str.compare("y_subdomain_sw"))
	{
		temp<<newvalue;
		temp>>y_subdomain_sw;
		return true;
	
		
	}
	else
	if(!str.compare("x_subdomain_ne"))
	{
		temp<<newvalue;
		temp>>x_subdomain_ne;
		return true;
	
		
	}
	else
	if(!str.compare("y_subdomain_ne"))
	{
		temp<<newvalue;
		temp>>y_subdomain_ne;
		return true;
	
		
	}
	else
	if(!str.compare("zo"))
	{
		temp<<newvalue;
		temp>>zo;
		return true;
	
		
	}
        else
	if(str.find("[")!=string::npos)  //This will handle the builidingsvector and then the subcases :
	{
		
              if(!(str.substr(0,str.find("["))).compare("buildings"))
		{

		  string s_index=str.substr(str.find("[")+1,str.find("]")-str.find("[")-1);
		 // cout<<"the index is :"<<s_index<<endl;
		  temp<<s_index;
                  int index;
                  temp>>index;
		 // cout<<"the integer value + 1 is"<<index+1<<endl;				   
		  //TODO : make sure there is no other text after the [] i.e only buildings[--] is allowed not buildings[1]a..
                  string building_varname=str.substr(str.find(".")+1,str.length()-1-str.find("."));
                 // cout<<"The varaible name is :"<<building_varname<<endl;
		


		//TODO: decide if resize required or not 
		temp.str("");
		temp.clear();   /////TODO: alternate way to do it /
		
		if(index+1>buildings.size())
                       buildings.resize(index+1);

		if(!building_varname.compare("group"))
		{
			
			temp<<newvalue;
			temp>>buildings[index].group;
           	   //  cout<<"value"<<value<<endl;
			return true;
		}
        	else
		if(!building_varname.compare("type"))
		{
			temp<<newvalue;
			temp>>buildings[index].type;
			return true;
	
		
		}
		else
		if(!building_varname.compare("height"))
		{
			temp<<newvalue;
			temp>>buildings[index].height;
			return true;
	
		
		}
		else
		if(!building_varname.compare("width"))
		{
			temp<<newvalue;
			temp>>buildings[index].width;
			return true;
	
		
		}
		else
		if(!building_varname.compare("length"))
		{
			temp<<newvalue;
			temp>>buildings[index].length;
			return true;
	
		
		}
		else
		if(!building_varname.compare("xfo"))
		{
			temp<<newvalue;
			temp>>buildings[index].xfo;
			return true;
	
		
		}
		else
		if(!building_varname.compare("yfo"))
		{
			temp<<newvalue;
			temp>>buildings[index].yfo;
			return true;
	
		
		}

		else
		if(!building_varname.compare("zfo"))
		{
			temp<<newvalue;
			temp>>buildings[index].zfo;
			return true;
	
		
		}
		else
		if(!building_varname.compare("gamma"))
		{
				//int value_temp;
			//cout<<"before changing:"<<buildings[index].gamma<<endl;
			//stringstream a;
			//a<<boost::any_cast<string>(newvalue);
			////a>>value_temp;
			//a>>buildings[index].gamma;
		
		
			temp<<newvalue;
			//cout<<"the string value is "<<value_temp<<endl;
			temp>>buildings[index].gamma;


		       // cout<<"the value changed by :"<<buildings[index].gamma<<endl;
			
			return true;
	
		
		}
		else
		if(!building_varname.compare("supplementalData"))
		{
			temp<<newvalue;
			temp>>buildings[index].supplementalData;
			return true;
	
		
		}
                else
		{
			cout<<"varaible not found:\t"<<building_varname<<endl;
	
		}








		  // temp<<boost::any_cast<string>(str.substr(str.);  
		  /*to allow int bldNum;
		  int group;
		  int type;
		  float height;
		  float width;
		  float length;
		  float xfo;
		  float yfo;
		  float zfo;
		  float gamma;
		  float supplementalData;*/



		//  cout<<"Trying to access the building vector :D "<<endl;
		}
		else 
                cout<<"wrong name of array :D "<<endl;
	
		
	}
	else
	{
		cout<<"Wrong data memeber"<<endl;
		
	}

}
~first(){}



int value;
float step;
std::string name;



};


#endif




