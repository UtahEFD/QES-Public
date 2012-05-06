/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Making drivers / stubs runnable from any directory.
*/

#ifndef DIRECTORY
#define DIRECTORY

#include <cstdlib>
#include <string>
#include <iostream>
#include <cassert>

void static inline getDirectories
(
 const int& argc, char* argv[], 
 std::string& inp_dir, std::string& out_dir,
 bool const beVerbose = false
)
{
	inp_dir = "./";
	out_dir = "./";

  assert(argc == 2 || argc == 3);
	inp_dir.assign(argv[1]);
	if (argc == 3)
	{
	  out_dir.assign(argv[2]);
  }
  else
  {
	  out_dir.assign("/tmp/");
  }
  
	//Check for '/' at end of directory strings.
	if(out_dir.at(out_dir.length() - 1) != '/') {out_dir = out_dir.append("/");}

  if (beVerbose)
  {
	  std::cout << std::endl;
	  std::cout << "<> Command line arguments <>" << std::endl;
	  std::cout << "Project Input  : " << inp_dir << std::endl;
	  std::cout << "Output directory: " << out_dir << std::endl;
  }
}

void static inline chdir_or_die(std::string dir)
{
 	if(chdir(dir.c_str()) != 0)
 	{
 		std::cerr << "Unable to change directory to " << dir << "." << std::endl;
 		std::cerr << "Exiting..." << std::endl;
 		exit(1);
 	}
}

void static inline chdir(std::string dir)
{
 	if(chdir(dir.c_str()) != 0)
 	{
 		std::cerr << "Unable to change directory to " << dir << "." << std::endl;
 	}
}

#endif
