/* File: PrefWindow.h
 * Author: Matthew Overby
 */

#ifndef SLUI_PREFWINDOW_H
#define SLUI_PREFWINDOW_H

#include "Button.h"
#include "Window.h"
#include "Scrollbar.h"
#include "OptionTracker.h"
#include "EventTracker.h"

namespace SLUI {

class PrefWindow : public Window {

	public:
		static const int Margin = 160;

		/** @brief BAD! Bad bad bad bad...
		*/
		PrefWindow(Window *win);

		/** @brief Constructor
		*/
		PrefWindow(int, WindowController*, OptionTracker*, EventTracker*, sf::RenderWindow*);

		/** @brief Default destructor
		*/
		~PrefWindow();

		/** @brief Checks to see what the mouse is over
		* 
		* This function loops through all of the buttons in the Buttons list,
		* calling each button's highlight function.
		*/
		void highlight();

		/** @brief Handle a mouse clicked event
		*/
		void mouseClicked();

		/** @brief Scrolls the text up or down based on the parameter
		* 
		* If you want to scroll up on the menu, set the parameter to true,
		* to scroll down, false.
		*/
		void scroll(bool up);

		/** @brief Changes the upper and lower bounds of a button
		*/
		void setButtonBounds(std::string command, float minVal, float maxVal);

		/** @brief Updates all the button visuals to their current value
		*/
		void update();

		/** @brief Adds a new button (regular/radio) to the menu window
		*/
		void addRadioButton(std::string, std::string);

		/** @brief Adds a new value button to the menu window
		*/
		void addValueButton(std::string, std::string, float);

		/** @brief Adds a new key button to the menu window
		*/
		void addKeyButton(std::string, std::string, std::string);

		/** @brief Adds a new list button to the menu window
		*/
		void addListButton(std::string, std::string, std::vector<std::string>);

		/** @brief Draws the preference window's elements
		*/
		void draw();

		/** @brief Resizes the preference window
		*/
		void resizeEvent();

		/** @brief Cleans up menu operations when the preferences window is closed
		* 
		* This function is called upon exiting the preferences window.  It unactivates
		* any key button that is currently waiting for a new key input
		*/
		void close();

		/** @brief Opens the Window
		*/
		void open();

		/** @brief Called in the event loop, checks to see if the event applies
		*/
		void checkEvent();

		bool newKeyListening;

	private:
		/** @brief Removes buttons off the top and bottom of the prefWindow
		*
		* This function finds the buttons that will fit into the given height,
		* starting at lineIndex, and populate the two lists buttonsLeft and
		* buttonsRight.
		*/
		void chopButtons();

		Button *closeButton;
		Button *applyButton;

		Scrollbar *scrollbar;
		OptionTracker *m_optTracker;

		std::map< std::string, Button*> leftButtons;
		std::map< std::string, Button*> rightButtons;

		std::string newKeyCommand;

		int leftButtonIndex;
		int rightButtonIndex;
		int leftButtonPosX;
		int rightButtonPosX;

		int lineIndex;
		int linesVisible;
		int linesTotal;
		int oldLinesTotal;
};

}

#endif

