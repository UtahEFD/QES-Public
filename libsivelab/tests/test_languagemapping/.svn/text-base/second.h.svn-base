#ifndef SECOND_H
#define SECOND_H 1
#include <boost/any.hpp>
#include <iomanip>
#include<iostream>
#include<string>
#include "base.h"
#include <sstream>
using boost::any_cast;
#include <map>
#include <vector>
//#include<ctype>
//using this to simulate the base class of quic data

class second : public base
{
public:


 enum ReleaseType
    {
      INSTANTANEOUS = 1,
      CONTINUOUS = 2,
      DISCRETE_CONTINUOUS = 3
    };
second(){

 x_subdomain_sw=0;
  y_subdomain_sw=0;
  x_subdomain_ne=0;
  y_subdomain_ne=0;

  zo=0;
	
 /*buildings.resize(5);
  buildings[0].type_enum=INSTANTANEOUS;
	///building the map

	  my_map["x_subdomain_sw"]=&x_subdomain_sw;
	  my_map["y_subdomain_sw"]=&y_subdomain_sw;
	  my_map["x_subdomain_ne"]=&x_subdomain_ne;
	  my_map["y_subdomain_ne"]=&y_subdomain_ne;
	  my_map["zo"]=&zo;
	  my_map["buildings[]"]=sizeof(buildingData);         //size of the structure 
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
	  my_map["buildings.type_enum"]=&buildings[0].type_enum;*/

build_map();
	
}

 int x_subdomain_sw;
  int y_subdomain_sw;
  int x_subdomain_ne;
  int y_subdomain_ne;

  float zo;


  struct buildingData
  {
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
    ReleaseType type_enum;
  };


  std::vector<buildingData> buildings;

void build_map(){
 buildings.resize(5);
  buildings[0].type_enum=INSTANTANEOUS;
	///building the map

	  my_map["x_subdomain_sw"]=&x_subdomain_sw;
	  my_map["y_subdomain_sw"]=&y_subdomain_sw;
	  my_map["x_subdomain_ne"]=&x_subdomain_ne;
	  my_map["y_subdomain_ne"]=&y_subdomain_ne;
	  my_map["zo"]=&zo;
	  my_map["buildings[]"]=sizeof(buildingData);         //size of the structure 
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
	  my_map["buildings.type_enum"]=&buildings[0].type_enum;

}
void eval_enum(boost::any enum_to_modify, std::string newvalue,int index,size_t size)
{
		char c;

		cerr<<"-===========-=-==============-=-=-=-=-=-=-=-=-=-=-=-=-=-=--===========-=-==============-=-=-=-=-=-=-=-=-=-=-=-=-=-=-"<<endl;
		cout<<"the value of enum before chaning "<<buildings[0].type_enum<<endl;

		//two ways to this again accept user input as string and do switch or if cases
                  // or take as integer type cast using the enumeration name and plug in the value like 	obj.type_enum=(second::ReleaseType)1;


		if(enum_to_modify.type()==typeid(ReleaseType *))
		{
		//TODO COMPLETE THIS STATEMENT
  			ReleaseType* temp_ptr=any_cast<ReleaseType*>(enum_to_modify);
	 		void * temp_voidptr=temp_ptr;
	 	 	temp_voidptr+=index*size;
                         stringstream temp(newvalue);
			int enum_value;
                         
			temp>>enum_value;
			if (!(temp.fail() || temp.get(c) || enum_value>3 ||enum_value<1 )) {
        			cout<<"the value being changed  "<<enum_value<<endl;			

			*(ReleaseType*)temp_voidptr=(ReleaseType)enum_value;
				// not an integer
       				}
			else
			{
				cerr<<enum_value<<"- is not an integer wrong working"<<endl;
    				return;
			}	
			
			//string temp_string;
			//temp>>temp_string;
	 		//*(ReleaseType*)temp_voidptr=DISCRETE_CONTINUOUS;



	cout<<"chagned the valuie size is "<<size<<endl;
		}



}
void print()
{

	if(buildings.size()<2)
		buildings.resize(5);
	  cout<<"\nthis is second class"<<endl;
	  cout<<"x_subdomain_sw:"<<x_subdomain_sw<<endl;
	  cout<<"y_subdomain_sw:"<<y_subdomain_sw<<endl;	
	   cout<<"x_subdomain_ne:"<<x_subdomain_ne<<endl;
	  cout<<"y_subdomain_ne:"<<y_subdomain_ne<<endl;
           cout<<"zo:"<<zo<<endl;

	cout<<"bldNum of buildings 0:\t"<<buildings[0].bldNum<<endl;
        cout<<"bldNum og buildigns 1:\t"<<buildings[1].bldNum<<endl;
	cout<<"xfo of builiding 0:\t"<<buildings[0].xfo<<endl;
	cout<<"xfo of builiding 1:\t"<<buildings[1].xfo<<endl;

        cout<<"the enum value is "<<buildings[0].type_enum<<endl;
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
   return true;


}
 bool modify_v2(std::string str,string newvalue)                          //should work for enums* :| should see how auto does this
{                                                                         //TODO make sure the else statmenets end the function rather than going forward
	stringstream temp;
       
	boost::any to_modify;
	 map<string,boost::any>::iterator it;
         int index=0;
         size_t size=0;
	if(str.find("[")!=string::npos)  //////////  all of this to handle vectors 
	{
		string array_name=str.substr(0,str.find("["));
		it=my_map.find(array_name+"[]");
		 if(it!=my_map.end())
		{
			  size=any_cast<size_t>(it->second);
		}		 
		else
		{
		
                                   cout<<"**********************************************"<<endl;
			           cout<<"un-identified array or vector name  in this class"<<endl;
                 		   cout<<"string not found"<<endl;
		}		
			
		
		string s_index=str.substr(str.find("[")+1,str.find("]")-str.find("[")-1);   ///gets the index of the vector or array
		temp.flush();
       		 temp.clear();

		 temp<<s_index;
                 
                  temp>>index;
		  				   
		  //TODO : make sure there is no other text after the [] i.e only buildings[--] is allowed not buildings[1]a..
                  string building_varname=str.substr(str.find(".")+1,str.length()-1-str.find("."));
		
		it=my_map.find(array_name+"."+building_varname);
		if(it==my_map.end())
                    cout<<"string not found"<<endl;
                else
		{
		to_modify=it->second;
		}
	}
        else                            ///this is the only case if it was without vectors :D
	{
		it=my_map.find(str);
		if(it==my_map.end())
		{	

			           cout<<"un-identified varaible in this class"<<endl;
                 		   cout<<"string not found"<<endl;
               
		}
		 else
		{
			to_modify=my_map[str];

		}
        }


       //***************this is necessary for conversion :| 
	temp.flush();
        temp.clear();

   	temp<<newvalue;
       int check;
	if( to_modify.type() == typeid(float*) ) 
	{

	  float* temp_ptr=any_cast<float*>(to_modify);
	  void * temp_voidptr=temp_ptr;
	  temp_voidptr+=index*size;
	  temp>>*(float*)temp_voidptr;
	
	//  temp>>*(any_cast<float*>(to_modify)+index*(size/sizeof(float)));
	}	
        else
        if(to_modify.type()==typeid(int*))
	{
	   temp>>*(any_cast<int*>(to_modify)+index*(size/sizeof(int)));
	}
	else
        if(to_modify.type()==typeid(ReleaseType *))
	{
		/*

		std::cout<<"**************************************************************** A MATCH*****************************"<<std::endl;
	   
		ReleaseType* temp_ptr=any_cast<ReleaseType*>(to_modify);
	 	void * temp_voidptr=temp_ptr;
	 	 temp_voidptr+=index*size;
		string temp_string;
		temp>>temp_string;
	 *(ReleaseType*)temp_voidptr=DISCRETE_CONTINUOUS;


	cout<<"chagned the valuie size is "<<size<<endl;
		*/

		string test_string;
				test_string =to_modify.type().name();
				if(test_string[test_string.length()-1]='E')
				{
					cout<<"is this a enum "<<str <<"  type_info  "<<test_string<<endl;
				}		




	}

return true;
}
~second(){}



int value;
float step;
std::string name;


};


#endif




