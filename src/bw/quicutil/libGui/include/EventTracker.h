/* File: EventTracker.h
 * Author: Matthew Overby
 */

#ifndef SLUI_EVENTCHECKER_H
#define SLUI_EVENTCHECKER_H

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>
#include <boost/function.hpp>
#include <boost/bind.hpp>

namespace SLUI {

namespace EventType {

	static const int W_Close = 0;
	static const int W_Resize = 1;
	static const int M_Scrollup = 2;
	static const int M_Scrolldown = 3;
	static const int M_Clickleft = 4;
	static const int M_Clickright = 5;
	static const int M_Dragleft = 6;
	static const int M_Dragright = 7;
	static const int K_Press = 8;
	static const int K_Released = 9;
	static const int M_Moved = 10;
}

class EventTracker {

	public:
		/** @brief Constructor
		*/
		EventTracker(sf::RenderWindow* app);

		/** @brief Destructor
		*/
		~EventTracker();

		void suppressWarnings( bool w ){ showWarnings = !w; }

		/* @brief See if the event is a regularly used event
		*
		* The list of Event tokens this function can return is in EventTracker.h
		*/
		void handleEvent(sf::Event *event);

		/** @brief Adds a function to the function list.
		* To call the function later, use "callFunction( label )"
		*/
		void addFunction( std::string label,  boost::function<void()> f );

		/** @brief Calls the function with the designated label
		*/
		void callFunction( std::string label );

		/** @brief Shows a string in the message center
		*
		* Displays a message (str) for a specified amount of time (t)
		* in seconds.  The message will be added to the console log.
		*/
		void showMessage(std::string str, int t);
		std::string message;
		int messageTime;

		int eventType;
		sf::Event *lastEvent;
		sf::Key::Code lastKeyCode;

		/** 
		* oldPos is set by the setMouseCoordinates function in Gui.cpp
		* newPos is set by the mouse__Dragged function on true
		*/
		sf::Vector2f oldPos, newPos;

	private:
		bool showWarnings;

		/** @brief Stores the table of functions
		*/
		std::map< std::string, boost::function<void()> > m_eventFuncs;

		/** @brief Checks to see if the left mouse button was dragged
		* @return true on mouse drag, false otherwise
		*
		* On mouse drag, stores the new mouse position as public
		* sf::Vector2f newPos
		*/
		bool mouseDragged();

		/** @brief Checks to see if the left mouse button was dragged
		* @return true on mouse drag, false otherwise
		*
		* On mouse drag, stores the new mouse position as public
		* sf::Vector2f newPos
		*/
		bool mouseLeftDragged();

		/** @brief Checks to see if the right mouse button was dragged
		* @return true on mouse drag, false otherwise
		*
		* On mouse drag, stores the new mouse position as public
		* sf::Vector2f newPos
		*/
		bool mouseRightDragged();

		sf::RenderWindow *m_App;

		/** @brief Stops possible function call loops
		* See the callFunction implementation for more details
		*/
		std::map< std::string, int > functionCount;

}; // end namespace EventTracker

} // end namespace SLUI

#endif
