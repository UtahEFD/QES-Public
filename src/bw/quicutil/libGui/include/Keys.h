/* File: Keys.h
 * Author: Matthew Overby
 */

#ifndef SLUI_KEYS_H
#define SLUI_KEYS_H

#include <SFML/Window.hpp>

namespace SLUI {

class Keys {

	public:
		/** @brief Default Constructor
		*/
		Keys();

		/** @brief Constructor
		* Creates a new key with the specified intial key
		*/
		Keys(std::string);

		/** @brief Default Destructor
		*/
		~Keys();

		/** @brief Get the string value of the key
		* @return std::string representation of the key code
		*/
		std::string toString();

		/** @brief Get the Key::Code value of the key
		* @return sf::Key::Code representation of the key code
		*/
		sf::Key::Code getKeyCode();

		/** @brief Changes the current key value
		* @return true if the new key is valid, false otherwise
		*/
		bool setNewKey(std::string);

		/** @brief sets this key to its default key code
		*/
		void toDefault();

		/** @brief: Returns the string value of an sf::Key::Code, "bad" if not found
		* @return: std::string representation of the key code
		**/
		std::string keyToString(unsigned int keycode);

		/** @brief: Returns the sf::Key::Code value of a string, escape key if not found
		* @return: sf::Key::Code representation of the key code
		**/
		sf::Key::Code findKey(std::string newKey);

	private:
		sf::Key::Code myKeyCode;
		std::string myKeyStr;
		sf::Key::Code defaultKeyCode;

};

}

#endif

