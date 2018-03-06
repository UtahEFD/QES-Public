#include "language_map.h"
std::string languageMap::retrieve(std::string variable_name){

///currWork todo 
	std::string var_type;
	void * temp_voidptr;

                   std::string current_value;
		   std::string inner_var_name;
		if(var_addressMap.empty())
		{	
				
			std::cerr<<"******************************************************************************************"<<std::endl;
			std::cerr<<"the map has not been intilaized because the build_map function was not overridden or build_map function was not called in the constructor"<<std::endl;	
			std::cerr<<"******************************************************************************************"<<std::endl;
			exit(1);
		}
		

		else
		{

                        
			//std::cerr<<"the map is not empty"<<std::endl;
			std::stringstream temp;
		       
			boost::any to_modify;   
			 std::map<std::string,boost::any>::iterator it;
			 int index=0;
			 size_t size=0;

			 if(variable_name.find(".")!=std::string::npos)    /// . is found 
                         {
				//std::cerr<<"the variable passed has a dot"<<variable_name<<std::endl;
				 inner_var_name=variable_name.substr(variable_name.find('.')+1,variable_name.length()-(variable_name.find('.')+1));
		               ///   std::cerr<<"the inner_var_name "<<inner_var_name<<std::endl;
				variable_name=variable_name.substr(0,variable_name.find('.'));
				//std::cerr<<"the variable name is"<<variable_name<<std::endl;
	 

				if(variable_name.find("[")!=std::string::npos)  //////////  all of this to handle vectors 
				{

					//std::cerr<<"trying to modify an array "<<std::endl;
					std::string array_name=variable_name.substr(0,variable_name.find("["));
					it=var_addressMap.find(array_name+"[]");
					 if(it!=var_addressMap.end())
					{
						  size=boost::any_cast<size_t>(it->second);
					}		 
					else
					{
		
						           std::cout<<"**********************************************"<<std::endl;
							  std::cout<<"un-identified array or vector name  in this class"<<std::endl;
					 		   std::cout<<"string not found"<<std::endl;
							  exit(1);
					}		
			
		
					std::string s_index=variable_name.substr(variable_name.find("[")+1,variable_name.find("]")-variable_name.find("[")-1);   ///gets the index of the vector or array

					//std::cerr<<"the index "<<s_index<<"the size "<<size<<std::endl;
					temp.flush();
			       		 temp.clear();

					 temp<<s_index;
					 
					  temp>>index;
					  if(index<0)
					    {

					      std::cout<<"The index cannot be less than zero "<<std::endl;
					      std::cout<<"varaible_name"<<variable_name<<std::endl;
					      std::cout<<"exiting*********************************"<<std::endl;
					      exit(1);
					    }
					  				   
					  //TODO : make sure there is no other text after the [] i.e only buildings[--] is allowed not buildings[1]a..

		                         ///TODO see if this works ?;
		                        // std::string building_varname=variable_name.substr(variable_name.find("]")+2,variable_name.length()-1-variable_name.find("."));       ///this makes sure there is nothing b/w buildings[10] and .
					  ///std::string building_varname=variable_name.substr(variable_name.find(".")+1,variable_name.length()-1-variable_name.find("."));
		


						///this is where there could be a inside vector. 
					///TODO : make it generic so that if there is a vector within a vector withn another one this would not work . Should make it generic so that it will work
					///here if the inner variable has a [] then its a vector again 
					if(inner_var_name.find("[")!=std::string::npos)
					{
					///the inner_var_name should have be "____[].____"
					std::string inner_array=inner_var_name.substr(0,inner_var_name.find("."));  //this should give me points[10]  
					std::string inner_index=inner_array.substr(inner_array.find("[")+1,inner_array.find("]")-inner_array.find("[")-1); //this should give me 10
					std::string inner_array_name = inner_array.substr(0,inner_array.find("["));
	
					std::string  var_to_check=variable_name +"."+ inner_array_name ;
					////sources[1].points;	

					//	std::cout<<"inner_array:"<<inner_array<<std::endl;
					//	std::cout<<"inner_index:"<<inner_index<<std::endl;
					//	std::cout<<"inner_array_name:"<<inner_array_name<<std::endl;
					//	std::cout<<"var_to_check:"<<var_to_check<<std::endl;
					
				//steps : 1) check to see if the new array exists . 
						it=var_addressMap.find(var_to_check);
						if(it!=var_addressMap.end())
						{
						  //	  std::cerr<<"before type_Cast vpoid"<<std::endl;			
						  ////TODO ::THIS would not work .So have to write a function that converts a given any to a void pointer . we use the pointers to get the diff and then we can add them
						  //// make sure that we make the additive term index the total value to add and the size to be 1 so that way we do not have th change the remaining logic 
						   temp_voidptr = boost::any_cast<void*>(it->second); //this should be the base address of the inner vector 
						  ///temp_ptr is the base address of the inner vector at a particulat index eg: sources[1].points
						  //   void *temp_ptr = boost::lexical_cast<void*>(it->second); //this should be the base address of the inner vector 
						  /// void * temp_ptr=it->second;
						   //	  std::cerr<<"after type_Cast vpoid"<<std::endl;			
						    	
		
						  //std::string size_inner_vector = array_name +"."+inner_array_name + "[]"; //this should be the size of inner vector
						  it=var_addressMap.find(array_name+"."+inner_array_name+"[]");
						  size_t size_inner_vector = boost::any_cast<size_t>(it->second);  ///size of the inner vector 
						    
						  // Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
						  // Need to test to make sure language still works.
						  temp_voidptr = (char *)temp_voidptr + (boost::lexical_cast<int>(inner_index))* size_inner_vector; //now this should be pointing to the base of any points vector
						    
						  //inner_var_name=variable_name.substr(variable_name.find('.')+1,variable_name.length()-(variable_name.find('.')+1));
						  std::string inner_var2_name = inner_var_name.substr(inner_var_name.find('.')+1,inner_var_name.length()-(inner_var_name.find('.')+1));
						  ///got the innervariable name now get the address. 
							
						  it=var_addressMap.find(array_name+"[0]."+inner_array_name);
						
						  void *inner_array_add=boost::any_cast<void*>(it->second);
						 
						  it=var_addressMap.find(array_name+"."+inner_array_name+"."+inner_var2_name);
						  //	std::cerr<<"the not found var"<<array_name<<"."<<inner_array_name<<"."<<inner_var2_name<<std::endl;
						  if(it==var_addressMap.end())
						    {
						      std::cout<<"this string not found"<<std::endl;
						      exit(1);	
						    }
						  else
						    {
						      //    std::cerr<<"It should break here:"<<std::endl;
							
						      void *inner_arr_var_add = return_type(it->second,var_type);  ///this is base address of smaller one
						      //    std::cerr<<"Broke "<<std::endl;
						    
						      
						      char *test1=(char*)inner_array_add;
						      char *test2=(char*)inner_arr_var_add;
						      size_t diff = test2-test1;

						      // Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
						      // Need to test to make sure language still works.
						      temp_voidptr = (char *)temp_voidptr + diff;

						      ///TODO STUCK HERE .

						      
						      //to_modify=
						      //now it should point to the location i want . 
						      //int * boo = (int*)(temp_ptr);
						      //  *boo=100;
						      
						      //    std::cerr<<"muhahah "<<diff;
						      
						    
						    }									
						
						  // std::cerr<<"omg"<<std::endl;
						  
			
					  }
						else
						{
							std::cout<<"string not found 3"<<std::endl;
							     exit(1);
						
						}
						
			


					}
					else
					{
							it=var_addressMap.find(array_name+"."+inner_var_name);
							//	std::cerr<<"not foundL"<<array_name<<"."<<inner_var_name<<std::endl;
							if(it==var_addressMap.end())
							{
							    std::cout<<"string not found 4"<<std::endl;
							     exit(1);
						
			
							}			
							else
							{
							to_modify=it->second;
							}
				
					}
				}
				else	
				{
				////this is for things where the variable is a class t
					//std::cout<<"this is for a class "<<std::endl;
					it=var_addressMap.find(variable_name);
					if(it==var_addressMap.end())
					{ 
						std::cout<<"un-identified varaible in this class"<<std::endl;
					 		   std::cout<<"string not found"<<std::endl;
								exit(1);

					}
					else
					{
						to_modify=var_addressMap[variable_name];
						void* temp_voidptr=boost::any_cast<void*>(to_modify);
						languageMap * base_ptr = (languageMap *) temp_voidptr;
						return base_ptr->retrieve(inner_var_name);
						 //to make sure this function ends here 
					}

				
				}



			

                         }////else direct varaiable type cast it to float etc
			else                            ///this is the only case if it was without vectors :D
			{
				//std::cout<<"This is a direct varaible without any . :"<<variable_name<<std::endl;
				it=var_addressMap.find(variable_name);
				if(it==var_addressMap.end())
				{	

						   std::cout<<"un-identified varaible in this class"<<std::endl;
				 		   std::cout<<"string not found"<<std::endl;
							exit(1);
			       
				}
				 else
				{
					to_modify=var_addressMap[variable_name];

				}
			}


		       //***************this is necessary for conversion :| 
			temp.flush();
			temp.clear();
				if( to_modify.type() == typeid(float*) || var_type.compare("float")==0 ) 
			{
			  //  std::cerr<<"i am here "<<std::endl;
	
			 // temp>>*(any_cast<float*>(to_modify)+index*(size/sizeof(float)));
	
		    //            std::cout<<"trying to modify the float number "<<std::endl;
			  if(var_type.compare("float")!=0)
			    {
			     
			        float* temp_ptr=boost::any_cast<float*>(to_modify);
	  			temp_voidptr=temp_ptr;

				// Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				// Need to test to make sure language still works.
	 			temp_voidptr = (char *)temp_voidptr + index*size;
			    }
			  //  std::cerr<<"the current value"<<*(float*)temp_voidptr<<std::endl;
			  current_value=boost::lexical_cast<std::string>(*(float*)temp_voidptr);
			  //
			  //	std::cerr<<"the current_Vale var:"<<current_value<<std::endl;

				//	std::cerr<<"this should not print"<<std::endl;


			}	
			else
			  if(to_modify.type()==typeid(int*)|| var_type.compare("int")==0)
			{
			  // temp>>*(any_cast<int*>(to_modify)+index*(size/sizeof(int)));
			  if(var_type.compare("int")!=0)
			    {

				int* temp_ptr=boost::any_cast<int*>(to_modify);
	  		        temp_voidptr=temp_ptr;

				// Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				// Need to test to make sure language still works.
	 			temp_voidptr = (char *)temp_voidptr + index*size;
			    }
			  current_value=boost::lexical_cast<std::string>(*(int*)temp_voidptr);
			}
			else 
			  if(to_modify.type()==typeid(double*)|| var_type.compare("double")==0)
			{
			  if(var_type.compare("double")!=0)
			    {
				double* temp_ptr=boost::any_cast<double*>(to_modify);
	  			temp_voidptr=temp_ptr;

				// Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				// Need to test to make sure language still works.
	 			temp_voidptr = (char *)temp_voidptr + index*size;
			    }
			  current_value=boost::lexical_cast<std::string>(*(double*)temp_voidptr);

			}
			else	
			  if(to_modify.type()==typeid(std::string*)|| var_type.compare("string")==0)
			{
			  if(var_type.compare("string")!=0)
			    {
				std::string* temp_ptr=boost::any_cast<std::string*>(to_modify);
	  			temp_voidptr=temp_ptr;

				// Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				// Need to test to make sure language still works.
	 			temp_voidptr = (char *)temp_voidptr + index*size;
			    }
			  current_value=boost::lexical_cast<std::string>(*(std::string*)temp_voidptr);



			}
			
			else 
			  if(to_modify.type()==typeid(short*) || var_type.compare("short")==0)
			{
			  if(var_type.compare("short")!=0)
			    {
				short* temp_ptr=boost::any_cast<short*>(to_modify);
	  			 temp_voidptr=temp_ptr;

				 // Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				 // Need to test to make sure language still works.
				 temp_voidptr = (char *)temp_voidptr + index*size;
			    }
			  current_value=boost::lexical_cast<std::string>(*(short*)temp_voidptr);



			}
			else 
			  if(to_modify.type()==typeid(long*)|| var_type.compare("long")==0)
			{
			  if(var_type.compare("long")!=0)
			    {
				long* temp_ptr=boost::any_cast<long*>(to_modify);
	  		        temp_voidptr=temp_ptr;

				// Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				// Need to test to make sure language still works.
	 			temp_voidptr = (char *)temp_voidptr + index*size;
			    }
			  current_value=boost::lexical_cast<std::string>(*(long*)temp_voidptr);



			}			
			else 
			  if(to_modify.type()==typeid(bool*)|| var_type.compare("bool")==0)
			{
			  if(var_type.compare("bool")!=0)
			    {
				bool* temp_ptr=boost::any_cast<bool*>(to_modify);
	  			temp_voidptr=temp_ptr;

				 // Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				 // Need to test to make sure language still works.
	 			temp_voidptr = (char *)temp_voidptr + index*size;
			    }
			  current_value=boost::lexical_cast<std::string>(*(bool*)temp_voidptr);



			}
			else
			{

				std::cout<<"type not found"<<std::endl;
				exit(1); ///this is to avoid the enum's for now
				std::string test_string;
				test_string =to_modify.type().name();
				/*	if(test_string[test_string.length()-1]='E')
				{
				  //	std::cout<<"is this a enum "<<variable_name <<std::endl;
					exit(1);
				//	eval_enum(to_modify,newvalue,index,size);
				}*/		

			
			

			}

			
		}
		return current_value;


}

 
void* languageMap::return_type(boost::any any_var,std::string& var_type)
{
/////
/////convert the boost::Any to a void pointer . Easy . 


	if( any_var.type() == typeid(float*) ) 
			{
	
			 // temp>>*(any_cast<float*>(to_modify)+index*(size/sizeof(float)));
	
		    //            std::cout<<"trying to modify the float number "<<std::endl;
				float* temp_ptr=boost::any_cast<float*>(any_var);
	  			void * temp_voidptr=temp_ptr;
				var_type="float";
	 			return temp_voidptr;
			




			}	
			else
			if(any_var.type()==typeid(int*))
			{
			  // temp>>*(any_cast<int*>(to_modify)+index*(size/sizeof(int)));

				int* temp_ptr=boost::any_cast<int*>(any_var);
	  			void * temp_voidptr=temp_ptr;
				var_type="int";
	 			return temp_voidptr;
			}
			else 
			if(any_var.type()==typeid(double*))
			{
				double* temp_ptr=boost::any_cast<double*>(any_var);
	  			void * temp_voidptr=temp_ptr;
	 			var_type="double";
	 			return temp_voidptr;
			}
			else	
			if(any_var.type()==typeid(std::string*))
			{
				std::string* temp_ptr=boost::any_cast<std::string*>(any_var);
	  			void * temp_voidptr=temp_ptr;
	 			var_type="string";
	 			return temp_voidptr;

			}
			else 
			if(any_var.type()==typeid(short*))
			{
				short* temp_ptr=boost::any_cast<short*>(any_var);
	  			void * temp_voidptr=temp_ptr;
	 			var_type="short";
	 			return temp_voidptr;


			}
			else 
			if(any_var.type()==typeid(long*))
			{
				long* temp_ptr=boost::any_cast<long*>(any_var);
	  			void * temp_voidptr=temp_ptr;
	 			var_type="long";
	 			return temp_voidptr;



			}			
			else 
			if(any_var.type()==typeid(bool*))
			{
				bool* temp_ptr=boost::any_cast<bool*>(any_var);
	  			void * temp_voidptr=temp_ptr;
	 			var_type="bool";
	 			return temp_voidptr;
	

			}
			else
			{

				std::cerr<<"We cannot modify anything but a basic type for a array within another array data member"<<std::endl;
				std::cerr<<"Breaking in language Mapping"<<std::endl;
				exit(1);

			}


	//	std::cerr<<"is it in the return fintion "<<std::endl;





}


void languageMap::modify_value(std::string variable_name,std::string newvalue)
{

	std::string var_type;
	void * temp_voidptr;
////changing the logic :
/* first check for a dot . 
if not found it should either a float char string or int 
if found . then its an array or a class
          if [ found then an array 
          else 
              if it a class with something in it. type cast it to a void * and then call language map on it so the inner one will be called*/
	//std::cout<<"INT MODIFY BVALYUE-------------------------------------:"<<variable_name<<" : "<<newvalue<<std::endl;
          
     //  std::cerr<<"Am i here - more testing"<<std::endl;
		std::string inner_var_name;///the name of the varaibale after the '.'

		if(var_addressMap.empty())
		{	
				
			std::cerr<<"******************************************************************************************"<<std::endl;
			std::cerr<<"the map has not been intilaized and  the build_map function was not overwritten hence we cannot access the datamembers"<<std::endl;	
			std::cerr<<"******************************************************************************************"<<std::endl;
			exit(1);
		}
		else
		{

                        
			//std::cerr<<"the map is not empty"<<std::endl;
			std::stringstream temp;
		       
			boost::any to_modify;   
			 std::map<std::string,boost::any>::iterator it;
			 int index=0;
			 size_t size=0;

			 if(variable_name.find(".")!=std::string::npos)    /// . is found 
                         {
			//	std::cerr<<"the variable passed has a dot"<<variable_name<<std::endl;
				 inner_var_name=variable_name.substr(variable_name.find('.')+1,variable_name.length()-(variable_name.find('.')+1));
		          //        std::cerr<<"the inner_var_name "<<inner_var_name<<std::endl;
				variable_name=variable_name.substr(0,variable_name.find('.'));
				//std::cerr<<"the variable name is"<<variable_name<<std::endl;
	 
		                   
				if(variable_name.find("[")!=std::string::npos)  //////////  all of this to handle vectors 
				{

					//std::cerr<<"trying to modify an array "<<std::endl;
					std::string array_name=variable_name.substr(0,variable_name.find("["));
					it=var_addressMap.find(array_name+"[]");
					 if(it!=var_addressMap.end())
					{
						  size=boost::any_cast<size_t>(it->second);
					}		 
					else
					{
		
						           std::cout<<"**********************************************"<<std::endl;
							  std::cout<<"un-identified array or vector name  in this class"<<std::endl;
					 		   std::cout<<"string not found"<<std::endl;
							  exit(1);
					}		
			
		
					std::string s_index=variable_name.substr(variable_name.find("[")+1,variable_name.find("]")-variable_name.find("[")-1);   ///gets the index of the vector or array

					//std::cerr<<"the index "<<s_index<<"the size "<<size<<std::endl;
					temp.flush();
			       		 temp.clear();

					 temp<<s_index;
					 
					  temp>>index;
					  if(index<0)
					    {

					      std::cout<<"The index cannot be less than zero "<<std::endl;
					      std::cout<<"varaible_name"<<variable_name<<std::endl;
					      std::cout<<"exiting*********************************"<<std::endl;
					      exit(1);
					    }
					  				   
					  //TODO : make sure there is no other text after the [] i.e only buildings[--] is allowed not buildings[1]a..

		                         ///TODO see if this works ?;
		                        // std::string building_varname=variable_name.substr(variable_name.find("]")+2,variable_name.length()-1-variable_name.find("."));       ///this makes sure there is nothing b/w buildings[10] and .
					  ///std::string building_varname=variable_name.substr(variable_name.find(".")+1,variable_name.length()-1-variable_name.find("."));
		


						///this is where there could be a inside vector. 
					///TODO : make it generic so that if there is a vector within a vector withn another one this would not work . Should make it generic so that it will work
					///here if the inner variable has a [] then its a vector again 
					if(inner_var_name.find("[")!=std::string::npos)
					{
					///the inner_var_name should have be "____[].____"
					std::string inner_array=inner_var_name.substr(0,inner_var_name.find("."));  //this should give me points[10]  
					std::string inner_index=inner_array.substr(inner_array.find("[")+1,inner_array.find("]")-inner_array.find("[")-1); //this should give me 10
					std::string inner_array_name = inner_array.substr(0,inner_array.find("["));
	
					std::string  var_to_check=variable_name +"."+ inner_array_name ;
					////sources[1].points;	

					//	std::cout<<"inner_array:"<<inner_array<<std::endl;
					//	std::cout<<"inner_index:"<<inner_index<<std::endl;
					//std::cout<<"inner_array_name:"<<inner_array_name<<std::endl;
					//std::cout<<"var_to_check:"<<var_to_check<<std::endl;
					
				//steps : 1) check to see if the new array exists . 
						it=var_addressMap.find(var_to_check);
						if(it!=var_addressMap.end())
						{
						  //	  std::cerr<<"before type_Cast vpoid"<<std::endl;			
						  ////TODO ::THIS would not work .So have to write a function that converts a given any to a void pointer . we use the pointers to get the diff and then we can add them
						  //// make sure that we make the additive term index the total value to add and the size to be 1 so that way we do not have th change the remaining logic 
						   temp_voidptr = boost::any_cast<void*>(it->second); //this should be the base address of the inner vector 
						  ///temp_ptr is the base address of the inner vector at a particulat index eg: sources[1].points
						  //   void *temp_ptr = boost::lexical_cast<void*>(it->second); //this should be the base address of the inner vector 
						  /// void * temp_ptr=it->second;
						   //  std::cerr<<"after type_Cast vpoid"<<std::endl;			
						    	
		
						  //std::string size_inner_vector = array_name +"."+inner_array_name + "[]"; //this should be the size of inner vector
						  it=var_addressMap.find(array_name+"."+inner_array_name+"[]");
						  size_t size_inner_vector = boost::any_cast<size_t>(it->second);  ///size of the inner vector 
						    
						  // Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
						  // Need to test to make sure language still works.
						  temp_voidptr = (char *)temp_voidptr + (boost::lexical_cast<int>(inner_index))* size_inner_vector; //now this should be pointing to the base of any points vector
						    
						  //inner_var_name=variable_name.substr(variable_name.find('.')+1,variable_name.length()-(variable_name.find('.')+1));
						  std::string inner_var2_name = inner_var_name.substr(inner_var_name.find('.')+1,inner_var_name.length()-(inner_var_name.find('.')+1));
						  ///got the innervariable name now get the address. 
							
						  it=var_addressMap.find(array_name+"[0]."+inner_array_name);
						
						  void *inner_array_add=boost::any_cast<void*>(it->second);
						 
						  it=var_addressMap.find(array_name+"."+inner_array_name+"."+inner_var2_name);
						  //	std::cerr<<"the not found var"<<array_name<<"."<<inner_array_name<<"."<<inner_var2_name<<std::endl;
						  if(it==var_addressMap.end())
						    {
						      std::cout<<"this string not found"<<std::endl;
						      exit(1);	
						    }
						  else
						    {
						      //      std::cerr<<"It should break here:"<<std::endl;
							
						      void *inner_arr_var_add = return_type(it->second,var_type);  ///this is base address of smaller one
						      // std::cerr<<"Broke "<<std::endl;
						    
						      
						      char *test1=(char*)inner_array_add;
						      char *test2=(char*)inner_arr_var_add;
						      size_t diff = test2-test1;
						      
						      // Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
						      // Need to test to make sure language still works.
						      temp_voidptr = (char *)temp_voidptr + diff;

						      ///TODO STUCK HERE .

						      
						      //to_modify=
						      //now it should point to the location i want . 
						      //int * boo = (int*)(temp_ptr);
						      //  *boo=100;
						      
						      //    std::cerr<<"muhahah "<<diff;
						      
						    
						    }									
						
						  // std::cerr<<"omg"<<std::endl;
						  
			
					  }
						else
						{
							std::cout<<"string not found 3"<<std::endl;
							     exit(1);
						
						}
						
			


					}
					else
					{
							it=var_addressMap.find(array_name+"."+inner_var_name);
							//		std::cerr<<"not foundL"<<array_name<<"."<<inner_var_name<<std::endl;
							if(it==var_addressMap.end())
							{
							    std::cout<<"string not found 4"<<std::endl;
							     exit(1);
						
			
							}			
							else
							{
							to_modify=it->second;
							}
				
					}
				}
				//std::cerr<<"sdafsdf"<<std::endl;
				else	
				{
				////this is for things where the variable is a class t
				//	std::cout<<"this is for a class "<<std::endl;
					it=var_addressMap.find(variable_name);
					if(it==var_addressMap.end())
					{ 
						std::cout<<"un-identified varaible in this class"<<std::endl;
					 		   std::cout<<"string not found"<<std::endl;
								exit(1);

					}
					else
					{
						to_modify=var_addressMap[variable_name];
						 temp_voidptr=boost::any_cast<void*>(to_modify);
						languageMap * base_ptr = (languageMap *) temp_voidptr;
						base_ptr->modify_value(inner_var_name,newvalue);
						return;   //to make sure this function ends here       ///this is where the object within another object is being handled
					}

				
				}



			

                         }
			 //	 std::cerr<<"zzz"<<std::endl;////else direct varaiable type cast it to float etc
			else                            ///this is the only case if it was without vectors :D
			{
			//	std::cout<<"This is a direct varaible without any . :"<<variable_name<<std::endl;
				it=var_addressMap.find(variable_name);
				if(it==var_addressMap.end())
				{	

						   std::cout<<"un-identified varaible in this class"<<std::endl;
				 		   std::cout<<"string not found"<<std::endl;
							exit(1);
			       
				}
				 else
				{
					to_modify=var_addressMap[variable_name];

				}
			}

			 //	 std::cerr<<"where "<<std::endl;
		       //***************this is necessary for conversion :| 
			temp.flush();
			temp.clear();
		   	temp<<newvalue;
                         		
			//	std::cerr<<"before zero "<<std::endl;
			if( to_modify.type() == typeid(float*) || var_type.compare("float")==0 ) 
			{
			  //			  std::cerr<<"i am here "<<std::endl;
	
			 // temp>>*(any_cast<float*>(to_modify)+index*(size/sizeof(float)));
	
		    //            std::cout<<"trying to modify the float number "<<std::endl;
			  if(var_type.compare("float")!=0)
			    {
			      //			      std::cerr<<"should not be here"<<std::endl;
			        float* temp_ptr=boost::any_cast<float*>(to_modify);
	  			temp_voidptr=temp_ptr;

				// Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				// Need to test to make sure language still works.
	 			temp_voidptr = (char *)temp_voidptr + index*size;
			    }
				//std::cout<<"before modification : inside the language map"<<*temp_ptr;
				//std::cout<<"the address in the lanaguge map is : "<<temp_voidptr<<std::endl;
			  //	  std::cerr<<"the address being modified"<<temp_voidptr<<std::endl;
			  //std::cerr<<"or is it here"<<std::endl;
	  			temp>>*(float*)temp_voidptr;
			

				//	std::cerr<<"this should not print"<<std::endl;


			}	
			else
			  if(to_modify.type()==typeid(int*)|| var_type.compare("int")==0)
			{
			  // temp>>*(any_cast<int*>(to_modify)+index*(size/sizeof(int)));
			  if(var_type.compare("int")!=0)
			    {

				int* temp_ptr=boost::any_cast<int*>(to_modify);
	  		        temp_voidptr=temp_ptr;

				// Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				// Need to test to make sure language still works.
	 			temp_voidptr = (char *)temp_voidptr + index*size;
			    }
	  			temp>>*(int*)temp_voidptr;
			}
			else 
			  if(to_modify.type()==typeid(double*)|| var_type.compare("double")==0)
			{
			  if(var_type.compare("double")!=0)
			    {
				double* temp_ptr=boost::any_cast<double*>(to_modify);
	  			temp_voidptr=temp_ptr;

				// Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				// Need to test to make sure language still works.
	 			temp_voidptr = (char *)temp_voidptr + index*size;
			    }
	  			temp>>*(double*)temp_voidptr;

			}
			else	
			  if(to_modify.type()==typeid(std::string*)|| var_type.compare("string")==0)
			{
			  if(var_type.compare("string")!=0)
			    {
				std::string* temp_ptr=boost::any_cast<std::string*>(to_modify);
	  			temp_voidptr=temp_ptr;

				// Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				// Need to test to make sure language still works.
	 			temp_voidptr = (char *)temp_voidptr + index*size;
			    }
	  			temp>>*(std::string*)temp_voidptr;



			}
			else 
			  if(to_modify.type()==typeid(short*) || var_type.compare("short")==0)
			{
			  if(var_type.compare("short")!=0)
			    {
				short* temp_ptr=boost::any_cast<short*>(to_modify);
	  			 temp_voidptr=temp_ptr;

				// Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				// Need to test to make sure language still works.
				 temp_voidptr = (char *)temp_voidptr + index*size;
			    }
	  			temp>>*(short*)temp_voidptr;



			}
			else 
			  if(to_modify.type()==typeid(long*)|| var_type.compare("long")==0)
			{
			  if(var_type.compare("long")!=0)
			    {
				long* temp_ptr=boost::any_cast<long*>(to_modify);
	  		        temp_voidptr=temp_ptr;

				// Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				// Need to test to make sure language still works.
	 			temp_voidptr = (char *)temp_voidptr + index*size;
			    }
				temp>>*(long*)temp_voidptr;



			}			
			else 
			  if(to_modify.type()==typeid(bool*)|| var_type.compare("bool")==0)
			{
			  if(var_type.compare("bool")!=0)
			    {
				bool* temp_ptr=boost::any_cast<bool*>(to_modify);
	  			temp_voidptr=temp_ptr;

				// Doing arithmetic on a void ptr get's complaints so one solution is to cast the void * to a char * to do the pointer math. --Pete, 10/23/2012
				// Need to test to make sure language still works.
	 			temp_voidptr = (char *)temp_voidptr + index*size;
			    }
	  			temp>>*(bool*)temp_voidptr;



			}
			else
			{

				std::cout<<"type not found"<<std::endl;
				exit(1); ///this is to avoid the enum's for now
				std::string test_string;
				test_string =to_modify.type().name();
				/*	if(test_string[test_string.length()-1]='E')
				{
					std::cout<<"is this a enum "<<variable_name <<std::endl;
					exit(1);
				//	eval_enum(to_modify,newvalue,index,size);
				}		
				*/
			
			

			}

			
		}





}
