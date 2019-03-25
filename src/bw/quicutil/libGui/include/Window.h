/* File: Window.h
 * Author: Matthew Overby
 *
 * NOTE:
 * Before implementing your own Window, you should
 * familiarize yourself with the WindowManager class
 * and other example window classes.  The process of
 * openning/closing, updating, and event checking isn't
 * exactly straight forward.  Most of the window handling
 * is done by WindowManager, including initialization. 
 */

#ifndef SLUI_WINDOW_H
#define SLUI_WINDOW_H

// TODO:  These should probably be moved to
//	  another file for easier editing.
#define QUIT 0
#define CONTROLS 1
#define VISUALS 2
#define GRAPHING 3
#define HELPGRAPH 4
#define HELPCONTROLS 5
#define HELPCONSOLE 6
#define CONSOLE 7
#define GRAPHWINDOW 8
#define CLICKCELLWINDOW 9
#define MENU 10

#include "Button.h"
#include "Widget.h"
#include "EventTracker.h"

namespace SLUI {

/* There is a shared WindowController that keeps track of a
*  global window stack located in Gui.h called m_winController.
*  This struct is placed here because all derivatives of Window
*  will need to use it to open and close themself.
*/
struct WindowController {

	/** @brief Pushes a window to the open stack
	*/
	void open(int window);

	/** @brief Removes a window from the open stack
	*/
	void close(int window);

	std::vector<int> m_openStack;
};

class Window : public Widget {

		/** The virtual functions of the Window base class
		*  are called by WindowManager
		*/

	public:
		/** @brief Display the window and elements
		*/
		virtual void draw() = 0;

		/** @brief Check to see if the event applies
		*/
		virtual void checkEvent() = 0;

		/** @brief Handle closing of the window
		*/
		virtual void close();

		/** @brief Handle opening of the window
		*/
		virtual void open();

		/** @brief Handle window mouseover highlights
		*/
		virtual void highlight();

		/** @brief Update any options or visuals
		* Update should be used to *retrieve* values from OptionTracker
		* and set buttons accordingly.
		*/
		virtual void update();

		/** @brief Handle a RenderWindow resize event
		*/
		virtual void resizeEvent();

		/** @brief Returns the ID of the window
		* @return int value
		*/
		int getId();

		/** @brief Used by TextWindow
		*/
		virtual void loadText( std::string file );

		/** @brief Used by ClickcellWindow
		*/
		virtual void setCellFuncLabel( std::string label );

	protected:
		EventTracker *m_eventTracker;
		WindowController *m_winController;
		int id;
};

}

#endif

