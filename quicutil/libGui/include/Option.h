/* File: Option.h
 * Author: Matthew Overby
 */

#ifndef SLUI_OPTION_H
#define SLUI_OPTION_H

#include <sstream>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cstdio>

namespace SLUI {

namespace OptionType {

	static const int Bool = 1;
	static const int Value = 2;
	static const int List = 3;
	static const int Flag = 4;
	static const int Time = 5;
}

class Option {

	public:
		/** @brief Returns the bool/value to its default setting
		*/
		virtual void setToDefault();

		/** @brief Toggles the active bool value member
		*
		* If the bool option's member "active" is currently true,
		* it changes it to false.  If it is false, it changes it to
		* true.  If this function is called, the bool member 
		* stateChanged is set to true.
		*/
		virtual void toggle();

		/** @brief Sets a new value to float/bool/list member value/active/string
		*
		* If the option is a bool option, you can set the value of active
		* to "true", "false", "on", or "off".  If it is a value option
		* you can set it to a float value.  It also flags stateChanged.
		*/
		virtual void setString(std::string);

		/** @brief Sets the option's float value to the new value
		 *
		 * setValue will also set stateChanged to true.
		 *
		 * @param value is a float value.
		 */
		virtual void setValue(float);

		/** @brief Returns the current value of the bool/value option
		* @return string representation of the option's current value
		*
		* If the option is a bool option, it will return "true" or "false".  
		* If it is a value option it will return the float value.
		*/
		virtual std::string getString();

		/** @brief Checks to see if value/active has been changed
		* @return true if the value/active member has been changed, false otherwise
		*
		* Use this function as a flag for the option's activity.  When you call 
		* stateChanged() it automatically sets the stateChanged bool value to false
		* and returns true if it was active.  The stateChanged bool value is set to 
		* true any time toggle(), setValue(), or setString() is called.
		*/
		bool stateChanged();

		/** @brief Returns the option's float value
		 * @return a float value
		 */
		virtual float getValue();

		/** @brief Sets the minimum and maxium a value can be
		* used by ValueOption
		 */
		virtual void setMinMax(float min, float max);

		int type;

	protected:
		bool stateChangedFlag;

};

class BoolOption : public Option {

	public:
		/** @brief Constructor bool
		* Creates a bool Option with intial value _init
		* Bool options can be referenced in the derived class
		* with the command "optTracker->getActive(<name>)".
		*/
		BoolOption( bool init );

		/** @brief Toggles the active bool value member
		* If this function is called, the bool member 
		* stateChanged is set to true.
		*/
		void toggle();

		/** @brief Sets a new value to the option
		*
		* You can set the string value to to "true", "false", "on", 
		* or "off".
		*/
		void setString(std::string val);

		/** @brief Returns the current value of the bool option
		* @return string representation of the option's current value
		* The string will be "true" or "false". 
		*/
		std::string getString();

		/** @brief Returns the bool to its default setting
		*/
		void setToDefault();

		/** @brief Returns the option's bool value
		 * @return a float value (0 or 1)
		 */
		float getValue();

		/** @brief Sets the option's boolean to a specific value.
		 * @param value should be 1 or 0.
		 */
		void setValue(float value);

	private:
		bool active, defaultActive;
};

class ValueOption : public Option {

	public:
		/** @brief Constructor value
		* Creates a value Option with intial value _init
		* Value options can be referenced in the derived class
		* with the command "optTracker->getValue(<name>)".
		*/
		ValueOption( float init );

		/** @brief Returns the value to its default setting
		* It also removes the min max bounds on the value
		*/
		void setToDefault();

		/** @brief Sets the minimum and maxium a value can be
		 */
		void setMinMax( float newMin, float newMax );

		/** @brief Sets a new value to the option
		*/
		void setString(std::string val);

		/** @brief Returns the current value of the value option
		* @return string representation of the option's current value
		*/
		std::string getString();

		/** @brief Returns the option's float value
		 * @return a float value
		 */
		float getValue();

		/** @brief Sets the option's float value to the new value
		 * setValue will also set stateChanged to true.
		 */
		void setValue(float newVal);

	private:
		bool bounded;
		float value, defaultValue;
		float min, max;
};

class ListOption : public Option {

	public:
		/** @brief Constructor list
		* Creates a list Option with intial array of options.
		* Currently selected options can be referenced in the derived class
		* with the command "optTracker->getListSelected(<name>)".
		*/
		ListOption( std::vector<std::string> init );

		/** @brief Sets the option's selected item to the new item
		 */
		void setString(std::string newValue);

		/** @brief Returns the option's current list value
		 * @return a string
		 */
		std::string getString();

		std::vector< std::string > listOptions;

	private:
		int index;
};

class TimeOption : public Option {

	public:
		/* @brief Constructor Time
		* Creates a Time option with the inital value init
		* Init should be a string of the forms
		*	HH:MM:SS (hour, minute, second)
		*	HH:MM
		*	HH
		* Where
		*	0 <= HH <= 24
		*	0 <= MM <= 59
		*	0 <= SS <= 59
		*/
		TimeOption( std::string init );

		/* @brief Set the time to its initial settings
		*/
		void setToDefault();

		/* @brief Set the time in the format HH:MM:SS
		*/
		void setString( std::string newValue );

		/* @brief Returns a string of the format HH:MM:SS
		*/
		std::string getString();

	private:
		int hour, default_hour;
		int minute, default_minute;
		int second, default_second;
};

}

#endif

