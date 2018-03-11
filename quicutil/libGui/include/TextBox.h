/* File: TextBox.h
 * Author: Matthew Overby
 *
 * NOTE:
 * This isn't the kind of textbox you can click on and
 * enter text.  It's meant for displaying text only.
 */

#ifndef SLUI_TEXTBOX_H
#define SLUI_TEXTBOX_H

#include <fstream>
#include "Widget.h"
#include "Scrollbar.h"

namespace SLUI {

class TextBox : public Widget {

	public:
		/** @brief Constructor
		*/
		TextBox();

		/** @brief Constructor
		*
		* Creates a new text box with the specified
		* width, height, x position and y position
		*/
		TextBox(int, int, int, int, sf::RenderWindow*);

		void setPosition(int posX, int posY);

		void setSize(int width, int height);

		/** @brief Destructor
		*/
		~TextBox();

		/** @brief Load text into the text box from a file
		*
		* Calling this function with the given text file location and name
		* will overwrite sf::String text and replace it with the contents of the file.
		* If the file is not found, the text will be "file not found".
		*/
		void loadText(std::string);

		/** @brief Sets the text to a new string
		*/
		void setText(std::string);

		/** @brief Returns a string value of all of the text
		* @return std::string value of the contents of the text box
		*/
		std::string getText();

		/** @brief Checks to see if the mouse is over the scroll bar
		* @return true if it is, false otherwise
		*/
		bool mouseOverScrollbar();

		/** @brief Sets the text format based off a copy of an sf::String
		*/
		void setTextFormat(sf::String);

		/** @brief Checks to see what the mouse is over
		* 
		* This function loops through all of the buttons, calling each button's 
		* highlight function.
		*/
		void highlight(const sf::Input *);

		/** @brief Checks to see what the mouse was clicked over
		* 
		* This function goes through all buttons and executes their function
		*/
		void mouseClicked(const sf::Input *);

		/** @brief Scrolls the text up or down based on the parameter
		* 
		* If you want to scroll up on the text, set the parameter to true,
		* to scroll down, false.
		*/
		void scroll(bool up);

		/** @brief Scrolls the text up or down based on the parameter
		* 
		* This is used when the bar is dragged and the parameter represents
		* the difference from old position to new position.
		*/
		void scroll(int diff);

		/** @brief Draws the text box
		*/
		void draw();

		void checkEvent();

	private:
		std::string fullText;
		sf::String text;
		float txtPadding;
		float scrollWidth;
		int lineIndex;
		int linesTotal;
		int linesVisible;

		Scrollbar scrollbar;

		/** @brief Removes beginning/ending of a string based on the height given.
		* @return A new sf::String
		*
		* The sf::String return will fit into the given height, and will start at
		* lineIndex (TextBox private variable)
		*/
		sf::String chopText(std::string, int);

		/** @brief Checks for file
		* @return true if the file exists, false otherwise
		* 
		* Checks to see if the text file is found, returns true
		* if it is, false otherwise
		*/
		bool checkFile(std::string);

};

}

#endif

