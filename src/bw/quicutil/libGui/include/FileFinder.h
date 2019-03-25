/* File: FileFinder.h
 * Author: Matthew Overby
 *
 * TODO:
 * Needs testing to make sure it's cross platform.
 */

#ifndef SLUI_FILEFINDER_H
#define SLUI_FILEFINDER_H

#include <sstream>
#include <iostream>
#include <boost/filesystem.hpp>

namespace SLUI {

class FileFinder {

	public:
		/** @brief Default Constructor
		* Sets base search directory in current (working) directory
		*/
		FileFinder();

		/** @brief Default Constructor
		* Sets base search directory to the one specified
		*/
		FileFinder( std::string baseDir );

		/** @brief Default Recursively search and find a file
		* Starting at the base directory, findRecursive searches
		* through all visible folders looking for the specified file.
		* @return Full path+filename to the file if found, blank string otherwise
		*/
		std::string findRecursive( std::string filename );

		/** @brief Default Search only base directory for a file
		* Limiting the search to the base directory, searches for the
		* specified file.
		* @return Full path+filename to the file if found, blank string otherwise
		*/
		std::string find( std::string filename );

	private:
		std::string baseDirectory;

};

}

#endif

