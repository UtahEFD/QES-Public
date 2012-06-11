#ifndef BASE_H
#define BASE_H 1
#include <boost/any.hpp>
#include <iomanip>
#include<iostream>
#include<string>
#include<map>
#include <sstream>
//using this to simulate the base class of quic data           





using namespace std;
using namespace boost;
class base
{
public:
base(){}
virtual boost::any retrieve(std::string str)=0;
  virtual void build_map(){}
virtual bool modify(std::string,boost::any)=0;   //assumes we pass int a correct typed variable i.e pass in int if it is an int
virtual bool modify_v2(std::string,std::string)=0;  //we pass in a string and then convert based on which type
virtual void eval_enum(boost::any , std::string,int index,size_t size)
{
	std::cerr<<"******************************************************************************************"<<std::endl;
			
	cerr<<"the enumeration eval function has not been overloaded"<<endl;
        cerr<<"the enumeration mentioned in the opt file cannot be changed"<<endl;

	std::cerr<<"******************************************************************************************"<<std::endl;


  /* or overload extraction operator like 
enum Sex {
    Male,
    Female
};

inline std::istream & operator>>(std::istream & str, Sex & v) {
  unsigned int sex = 0;
  if (!(str >> sex))
    return str;
  if (sex >= Female) {
    str.setstate(str.rdstate() | std::ios::failbit);
    return str;
  }
  v = static_cast<Sex>(sex);
  return str;
}*/


}
virtual void modify_value(std::string str,std::string newvalue)
	{
	
		if(my_map.empty())
		{	
				
			std::cerr<<"******************************************************************************************"<<std::endl;
			std::cerr<<"the map has not been intilaized and so the build_map function was not overwritten "<<std::endl;	
			std::cerr<<"******************************************************************************************"<<std::endl;

		}
		else
		{

			//std::cerr<<"the map is not empty"<<std::endl;
			std::stringstream temp;
		       
			boost::any to_modify;
			 std::map<std::string,boost::any>::iterator it;
			 int index=0;
			 size_t size=0;
			if(str.find("[")!=std::string::npos)  //////////  all of this to handle vectors 
			{
				std::string array_name=str.substr(0,str.find("["));
				it=my_map.find(array_name+"[]");
				 if(it!=my_map.end())
				{
					  size=any_cast<size_t>(it->second);
				}		 
				else
				{
		
				                   std::cout<<"**********************************************"<<std::endl;
						  std::cout<<"un-identified array or vector name  in this class"<<std::endl;
				 		   std::cout<<"string not found"<<std::endl;
                                                   return ;
				}		
			
		
				std::string s_index=str.substr(str.find("[")+1,str.find("]")-str.find("[")-1);   ///gets the index of the vector or array
				temp.flush();
		       		 temp.clear();

				 temp<<s_index;
				 
				  temp>>index;
				  				   
				  //TODO : make sure there is no other text after the [] i.e only buildings[--] is allowed not buildings[1]a..

                                 ///TODO see if this works ?
                                 std::string building_varname=str.substr(str.find("]")+2,str.length()-1-str.find("."));       ///this makes sure there is nothing b/w buildings[10] and .
				  ///std::string building_varname=str.substr(str.find(".")+1,str.length()-1-str.find("."));
		
				it=my_map.find(array_name+"."+building_varname);
				if(it==my_map.end())
				{
				    std::cout<<"string not found"<<std::endl;
				
					return;
			
				}			
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

						   std::cout<<"un-identified varaible in this class"<<std::endl;
				 		   std::cout<<"string not found"<<std::endl;
							return ;
			       
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
		
			if( to_modify.type() == typeid(float*) ) 
			{
	
			 // temp>>*(any_cast<float*>(to_modify)+index*(size/sizeof(float)));
	
		
				float* temp_ptr=any_cast<float*>(to_modify);
	  			void * temp_voidptr=temp_ptr;
	 			temp_voidptr+=index*size;
	  			temp>>*(float*)temp_voidptr;




			}	
			else
			if(to_modify.type()==typeid(int*))
			{
			  // temp>>*(any_cast<int*>(to_modify)+index*(size/sizeof(int)));

				int* temp_ptr=any_cast<int*>(to_modify);
	  			void * temp_voidptr=temp_ptr;
	 			temp_voidptr+=index*size;
	  			temp>>*(int*)temp_voidptr;
			}
			else
			if(to_modify.type()==typeid(double*))
			{
				double* temp_ptr=any_cast<double*>(to_modify);
	  			void * temp_voidptr=temp_ptr;
	 			temp_voidptr+=index*size;
	  			temp>>*(double*)temp_voidptr;

			}
			else	
			if(to_modify.type()==typeid(string*))
			{
				string* temp_ptr=any_cast<string*>(to_modify);
	  			void * temp_voidptr=temp_ptr;
	 			temp_voidptr+=index*size;
	  			temp>>*(string*)temp_voidptr;



			}
			else
			{
			
				string test_string;
				test_string =to_modify.type().name();
				if(test_string[test_string.length()-1]='E')
				{
					cout<<"is this a enum "<<str <<endl;

					eval_enum(to_modify,newvalue,index,size);
				}		

			
			

			}

			
		}

	


	}
virtual void print()=0;
~base(){}

protected:
std::map<std::string,boost::any>  my_map;   								//change's made end




};


#endif
