#include "first.h"
#include "second.h"
#include <map>
#include <fstream>
#include <sstream>
using namespace std;
///TODO: some problem with the while loop its reading the last line twice :| :| 
map<string,base *> mapping;

///TODO : ready with the string parse and point and then pass the required string to modify_v2

void update_with_file()
{
	std::ifstream testFile;
  	testFile.open("test_input.txt");
	map<string,base*>::iterator it;
	
	char line[1024];
        std::string wholeline;
  	std::string classname;
        std::string datamember;
        std::string value;
        
  	while(!testFile.eof())
    	{
      		testFile.getline(line, 1024);
		stringstream holder(line);

		holder>>wholeline;
	//	cout<<"-- " <<wholeline;
	

		//checking for access to class
		 size_t found;
		found=wholeline.find("."); 
		
		if(found!=string::npos)
		{
			
		   classname=wholeline.substr(0,found);
			//cout<<"\n so the class names are "<<classname<<endl;
		

		 	it=mapping.find(classname);

			
			if(it!=mapping.end())
			{
				
				//string x="adfasdfasdf";
				datamember=wholeline.substr(wholeline.find('.')+1,wholeline.length()-(wholeline.find('.')-1));
				//cout<<"the datamember is "<<datamember<<endl;
				holder>>value;
				it->second->modify_value(datamember,value);
			
			}	
		
			else
			{
				cout<<"\n Specified class not found"<<endl;
			}	

		}
		else
		{
			cout<<"error with syntax"<<endl;

		}		
		
	}


		
}
int main()
{


   int i_ans=0;
   float f_ans=0.0;
   string s_ans="";
	
  mapping["first"]=new first();
   mapping["second"]=new second();
	

	cout<<"\nvalues before using file to update"<<endl;	
	cout<<"-------------------------------------"<<endl;
	mapping["first"]->print();
	mapping["second"]->print();
		update_with_file();
  	cout<<"\nvalues after using file to update"<<endl;	
	cout<<"-------------------------------------"<<endl;
	mapping["first"]->print();
	mapping["second"]->print();
	
  /*
  base *ptr,*ptr1;
  ptr=new first(10,2.3,"hell");
  ptr1=new second(9,9.9,"blaze");
  
	boost::any value=ptr1->retrieve("name");
 
	 if(value.type() == typeid(int))
	 {
	   i_ans=boost::any_cast<int>(value);
	 }
	 else
	 if(value.type() == typeid(float))
	 { 
	  f_ans=boost::any_cast<float>(value);
	 }
	 else
	 if(value.type() == typeid(string))
	 { 
	  s_ans=boost::any_cast<string>(value);
	 }

cout<<"the values arfe "<<i_ans<<" "<<f_ans<<" "<<s_ans<<endl;

string x="dfasdf";
     cout<<"worked "<<ptr->modify_v2("name",x);
//for the second pointer 
	value=ptr->retrieve("name");
 
	 if(value.type() == typeid(int))
	 {
	   i_ans=boost::any_cast<int>(value);
	 }
	 else
	 if(value.type() == typeid(float))
	 { 
	  f_ans=boost::any_cast<float>(value);
	 }
	 else
	 if(value.type() == typeid(string))
	 { 
	  s_ans=boost::any_cast<string>(value);
	 }


cout<<"the values arfe "<<i_ans<<" "<<f_ans<<" "<<s_ans<<endl;

 
	

*/
return 0;
}
