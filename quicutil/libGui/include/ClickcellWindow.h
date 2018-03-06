/* File: ClickcellWindow.h
 * Author: Matthew Overby
 *
 * NOTE:
 * This class isn't meant to be used for your own purposes.
 * If you look at the implementation file you'll see references
 * to options that were never initialized in Gui.cpp.
 * You should really only use it as a reference on how to
 * interact with and use clickcells.
 */

#ifndef SLUI_CLICKCELLWINDOW_H
#define SLUI_CLICKCELLWINDOW_H

#include "Button.h"
#include "Widget.h"
#include "TextBox.h"
#include "Window.h"
#include "OptionTracker.h"
#include "EventTracker.h"
#include "CellTracker.h"

namespace SLUI {

class ClickcellWindow : public Window {

	public:
		/** @brief Constructor
		*/
		ClickcellWindow(int, WindowController*, CellTracker*, OptionTracker*, EventTracker*, sf::RenderWindow*);

		/** @brief Destructor
		*/
		~ClickcellWindow();

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

		/** @brief Draws the window's elements
		*/
		void draw();

		/** @brief Updates the viewable clickcell's stats
		*/
		void update();

		/** @brief Resizes the window
		* 
		* When the App window is resized, this must be called with the
		* new height and width to properly adjust its dimensions
		*/
		void resizeEvent();

		/** @brief Cleans up menu operations when the window is closed
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
		Button *cellButton;
		Button *refreshButton;
		Button *showButton;
		Button *propButton;
		Button *saveButton;

		TextBox textBox;
		OptionTracker *m_optTracker;
		CellTracker *m_cellTracker;

		sf::Color defaultColor;
		sf::Color menuItemDown;

};

}

#endif

