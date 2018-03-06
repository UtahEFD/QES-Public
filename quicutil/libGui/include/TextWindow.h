/* File: TextWindow.h
 * Author: Matthew Overby
 */

#ifndef SLUI_TEXTWINDOW_H
#define SLUI_TEXTWINDOW_H

#include "Button.h"
#include "Widget.h"
#include "TextBox.h"
#include "Window.h"
#include "OptionTracker.h"
#include "EventTracker.h"

namespace SLUI {

class TextWindow : public Window {

	public:
		static const int Margin = 160;

		/** @brief Constructor
		*/
		TextWindow(int, WindowController*, OptionTracker*, EventTracker*, sf::RenderWindow*);

		/** @brief Constructor
		*/
		~TextWindow();

		/** @brief Load text into the text box from a file
		*
		* Calling this function with the given text file location and name
		* will overwrite sf::String text and replace it with the contents of the file.
		* If the file is not found, the text will be "file not found".  It does not
		* use the FileFinder class, which is what should be used before this function
		* is called.
		*
		* The file name will be stored and reused on a window resize event.
		*/
		void loadText(std::string);

		/** @brief Checks to see what the mouse is over and highlight buttons if needed
		* 
		* This function loops through all of the buttons in the Buttons map table,
		* calling each button's highlight function to highlight.
		*/
		void highlight();

		/** @brief Checks to see what the mouse is over
		* @return ButtonInfo structure of the button.
		* 
		* This function loops through all of the buttons in the Buttons map table,
		* calling each button's mouseOver function.  If it is over a button,
		* returns that button's information.
		*/
		void mouseClicked();

		/** @brief Scrolls the text up or down based on the parameter
		* 
		* If you want to scroll up on the text, set the parameter to true,
		* to scroll down, false.
		*/
		void scroll(bool up);

		/** @brief Draws the text window's elements
		*/
		void draw();

		/** @brief Resizes the text window
		* 
		* When the App window is resized, this must be called with the
		* new height and width to properly adjust its dimensions
		*/
		void resizeEvent();

		/** @brief Cleans up menu operations when the text window is closed
		*/
		void close();

		/** @brief Opens the Window
		*/
		void open();

		/** @brief Called in the event loop, checks to see if the event applies
		*/
		void checkEvent();

	private:
		Button *closeButton;

		TextBox textBox;
		std::string txtFile;
		OptionTracker *m_optTracker;

		sf::Color defaultColor;
		sf::Color menuItemDown;

};

}

#endif

