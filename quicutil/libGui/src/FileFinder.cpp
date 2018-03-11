/* File: FileFinder.cpp
 * Author: Matthew Overby
 */

#include "FileFinder.h"

using namespace SLUI;
using namespace std;


FileFinder::FileFinder(){ baseDirectory = "."; }


FileFinder::FileFinder( string baseDir ){ baseDirectory = baseDir; }


string FileFinder::findRecursive( string filename ){

	string result = "";

	boost::filesystem::recursive_directory_iterator file_end;
	boost::filesystem::recursive_directory_iterator iter( baseDirectory );

	// Recursively loop through directories looking for the file
	for( iter; iter != file_end; iter++ ){
		if( filename.compare( iter->path().filename().c_str() ) == 0 ){
			result = complete(*iter).string();
		} // end file found
	} // end iterate through all files

	return result;

} // end recursive find


string FileFinder::find( string filename ){

	string result = "";

	boost::filesystem::directory_iterator file_end;
	boost::filesystem::directory_iterator iter( baseDirectory );

	// Recursively loop through directories looking for the file
	for( iter; iter != file_end; iter++ ){
		if( filename.compare( iter->path().filename().c_str() ) == 0 ){
			result = complete(*iter).string();
		} // end file found
	} // end iterate through all files

	return result;

} // end find file
