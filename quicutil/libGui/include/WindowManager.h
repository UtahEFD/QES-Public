/* File: WindowManager.h
 * Author: Matthew Overby
 *
 * TODO:
 * There should be no reliance on window types when it comes
 * to the calls in WindowManager.
 */

#ifndef SLUI_WINDOWMANAGER_H
#define SLUI_WINDOWMANAGER_H

#include "OptionTracker.h"
#include "Console.h"
#include "Window.h"
#include "FileFinder.h"
#define CONSOLE 7

namespace SLUI {

class WindowManager {

	public:
		/** @brief Constructor
		*/
		WindowManager(WindowController *wnC, EventTracker *evt, OptionTracker *opT, sf::RenderWindow *app);

		/** @brief Destructor
		*/
		~WindowManager();

		/** @brief Add a new window to the WindowManager
		* Memory deallocation is handled in the WindowManager destructor.
		*/
		void addWindow(Window* newWin);

		/** @brief Pushes a window to the open stack
		*/
		void open(int);

		/** @brief Removes a window from the open stack
		*/
		void close(int);

		/** @brief Checks to see if any windows are open
		* @return true if a window is open, false otherwise
		*/
		bool isOpen();

		/** @brief Checks to see if the window is on top
		* @return true if a window is top most, false otherwise
		*/
		bool isTop(int);

		/** @brief Checks to see if a specific window is open
		* @return true if the window is open, false otherwise
		*/
		bool isOpen(int);

		/** @brief Get the window that was most recently open
		* @return int value of that window
		*/
		int getTop();

		/** @brief Draws all open windows (refers to WindowTracker)
		*/
		void drawWindows();

		/** @brief Calls every window's checkEvent when applicable
		*/
		void checkEvent();

		/** @brief Updates the PrefWindow's options
		*/
		void update();

		/** @brief Updates a specific window
		*/
		void update(int id);

		/** @brief Get a specific window with given ID
		* @return a pointer to a window object
		*/
		Window* getWindow(int id);

		/** @brief Loads a text file into a TextWindow
		*/
		void loadText(int id, std::string file);

	private:
		OptionTracker *m_optTracker;
		WindowController *m_winController;
		EventTracker *m_eventTracker;
		Console *m_console;
		sf::RenderWindow *m_App;

		std::map<int, Window*> m_winTable;

};

}

#endif

