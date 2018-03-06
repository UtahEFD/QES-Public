/* File: OptionTracker.h
 * Author: Matthew Overby
 */

#ifndef SLUI_OPTIONTRACKER_H
#define SLUI_OPTIONTRACKER_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <map>
#include "Keys.h"
#include "Option.h"

namespace SLUI {

class OptionTracker {

	public:
		/** @brief Default Constructor
		*/
		OptionTracker();

		/** @brief Default Destructor
		*/
		~OptionTracker();

		/** @brief Adds a new option
		*/
		void addOption( std::string command, Option *newOption );

	/**********
	*	Bool Options
	***********/

		/** @brief Add a bool option
		* @params Index name and initial value
		*/
		void addBoolOption(std::string, bool);

		/** @brief Toggles the bool value of the given option
		*/
		void toggle(std::string);

		/** @brief Returns the bool value of the given option
		* @return a bool value
		*/
		bool getActive(std::string);

		/** @brief Sets the bool value of the given option
		*/
		void setActive(std::string, bool);

	/**********
	*	Value Options
	***********/

		/** @brief Add a value option
		* @params Index name and initial value
		*/
		void addValueOption(std::string, float);

		/** @brief Returns the float value of the given option
		* @return a float value
		*/
		float getValue(std::string);

		/** @brief Sets the float value of the given option
		*/
		void setValue(std::string, float);

		/** @brief Sets minimum and maximum the value can be
		*/
		void setMinMax(std::string, float min, float max);


	/**********
	*	Bool/Value Options
	***********/

		/** @brief Returns a string representation of the option's value
		* @return an std::string
		*/
		std::string getString(std::string);

		/** @brief Sets the option's value given a string
		*
		* This function will convert a string representation of a number
		* or bool value and set it to the corresponding option.
		*/
		void setString(std::string, std::string);

		/** @brief Checks to see if an option has been changed
		*
		* Look at Option::stateChanged() for more information
		*/
		bool stateChanged(std::string);

		/** @brief Get a copy of the list of options
		* @return Map table of current options
		*/
		std::map<std::string, Option*> getOptions();

		/** @brief Get a copy of the list of keys
		* @return Map table of current keys
		*/
		std::map<std::string, Keys*> getKeys();

	/**********
	*	List Options
	***********/

		/** @brief Add a new list option
		* @params Index name and list of options
		*/
		void addListOption(std::string, std::vector< std::string > opts);

		/** @brief Returns the option's current list value
		 * @return a string
		 */
		std::string getListValue(std::string command);

		/** @brief Sets the option's list value
		 */
		void setListValue(std::string command, std::string value);

	/**********
	*	Key Options
	***********/

		/** @brief Add a key option
		* @params Index name and initial key
		*/
		void addKeyOption(std::string, std::string);

		/** @brief Returns the Key::Code of the given key
		* @return an sf::Key::Code value
		*/
		sf::Key::Code getKey(std::string);

		/** @brief Returns the string representation of the given key
		* @return an std::string
		*/
		std::string getKeyStr(std::string);

		/** @brief Binds a new key to Keys table element
		* @return true if key is successfully bound, false otherwise
		*
		* For a list of bindable keys, see Keys.h
		* Attemps to bind (desiredKey) to element Keys with 
		* map-key (command).  If the new key is successfully
		* set, it will return true
		*/
		bool bind(std::string, std::string);

	private:
		std::map<std::string, Option*> options;
		std::map<std::string, Keys*> keys;

};

}

#endif

