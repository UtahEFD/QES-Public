/* File: Console.h
 * Author: Matthew Overby
 *
 * TODO:
 * Re-implement with boost::regex!  The current
 * scheme of nestest if statements is aweful.
 */

#ifndef SLUI_CONSOLE_H
#define SLUI_CONSOLE_H

#include "TextBox.h"
#include "EventTracker.h"
#include "OptionTracker.h"
#include "Window.h"

namespace SLUI {

struct Message {

	/** @brief Constructor
	*/
	Message(sf::RenderWindow *app);

	/** @brief Displays the message for a certain time (displayTime)
	*/
	void draw();

	sf::String text;
	sf::Clock clock;
	sf::RenderWindow *m_App;
	int displayTime;
	bool show;
};

class Console : public Window {

	public:
		/** @brief Constructor
		*/
		Console(int newId, WindowController *wnC, OptionTracker *opT, EventTracker *evt, sf::RenderWindow* app);

		/** @brief Destructor
		*/
		~Console();

		/** @brief Draws all elements associated with console
		*/
		void draw();

		/** @brief Called when the console is hidden for cleanup
		*/
		void close();

		/** @brief Called when the console is opened
		*/
		void open();

		/** @brief Sets the new character to the current input
		*/
		void addInput(char newChar);

		/** @brief Removes the current input 
		*/
		void clearInput();

		/** @brief Shows a string in the message center
		*
		* Displays a message (str) for a specified amount of time (t)
		* in seconds.  The message will be added to the console log.
		* Don't call this message directly.  Instead, call the showMessage
		* function in EventTracker.
		*/
		void showMessage(std::string str, int t);

		/** @brief Returns the line of text entered
		* @return string value of the current input 
		*/
		std::string getInput();

		/** @brief Called in the event loop, checks to see if the event applies
		*
		* Calls the functions
		* - highlight
		* - mouseClicked
		* - handleKeyPressed
		* - enterInput
		*/
		void checkEvent();

		/** @brief Resizes the console on a resize event
		*/
		void resizeEvent();

	private:
		/** @brief Handles if the event was a key press
		*/
		void handleKeyPressed();

		/** @brief Adds specified string to the log
		*/
		void enterInput(std::string str);

		/** @brief Checks if mouse is over buttons, highlights them if it is
		*/
		void highlight();

		/** @brief Checks to see what the mouse was clicked over
		* 
		* This function goes through all buttons and executes their functions
		*/
		void mouseClicked();

		/** @brief Scrolls the text up or down based on the parameter
		* 
		* If you want to scroll up on the text, set the parameter to true,
		* to scroll down, false.
		*/
		void scroll(bool up);

		sf::String input;
		sf::Sprite inputBar;
		TextBox textBox;
		sf::String logStyle;
		OptionTracker *m_optTracker;
		EventTracker *m_eventTracker;
		Message *m_message;
		sf::Shape messageBg;

		bool isOpen;
};

}

#endif

