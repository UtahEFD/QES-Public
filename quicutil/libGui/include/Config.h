/* File: Config.h
 * Author: Matthew Overby
 */

#ifndef SLUI_CONFIG_H
#define SLUI_CONFIG_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include "OptionTracker.h"

namespace SLUI {

class Config {

	public:
		/** @brief Default Constructor
		*/
		Config(OptionTracker* opt);

		/** @brief Default Destructor
		*/
		~Config();

		/** @brief Loads the configuration file
		* @return true if the config file successfully loaded, false otherwise
		* 
		* If the config file exists, this method loops through and parses line by line.
		* It will load the values in the config file and set them to the specified
		* option/key.
		*/
		bool loadConfig();

		/** @brief Updates the configuration file
		* 
		* If the configuration file is not found, it will create a default one.
		* Then, it will write all options in m_optTracker to the file.
		*/
		void updateConfig();

	private:
		/** @brief Checks for file
		* @return true if the file exists, false otherwise
		*/
		bool checkFile(std::string filename);

		OptionTracker* m_optTracker;
};

}

#endif

