/* File: MenuTracker.h
 * Author: Matthew Overby
 *
 * TODO:
 * Fix the whole "mousemenu" thing that's kind of become
 * a problem.  It was hastily implemented and needs
 * revision.
 */

#ifndef SLUI_MENUTRACKER_H
#define SLUI_MENUTRACKER_H

#include "WindowManager.h"
#include "EventTracker.h"
#include "OptionTracker.h"
#include "MenuItem.h"

namespace SLUI {

class MenuTracker {

	public:
		/** @brief Default Constructor
		*/
		MenuTracker(WindowManager *m_w, EventTracker *m_e, OptionTracker *m_o, sf::RenderWindow *m_a);

		/** @brief Default Destructor
		*/
		~MenuTracker();

		/** @brief Create a new menu item (group)
		* New MenuItems will be shown in the order they are created
		*/
		void createMenu( std::string label );

		/** @brief Create a new menu item (group) from a list option
		* New MenuItems will be shown in the order they are created.
		*/
		void createMenu( std::string label, std::string command, std::vector<std::string> opts );

		/** @brief Add a window item to a menu button
		*/
		void addItem(int WINDOW, std::string sub_label, std::string menu_label);

		/** @brief Add a flag item to a menu button
		*/
		void addItem(std::string command, std::string sub_label, std::string menu_label);

		/** @brief Draws the menu background and all menu items
		*/
		void draw();

		/** @brief Called in the event loop, checks to see if the event applies
		*/
		void checkEvent();

		/** @brief Checks to see if any of the menu items are dropped
		*@return True is any one is, false otherwise
		*/
		bool isOpen();

		MenuItem m_mouseMenu;

	private:
		WindowManager *m_winTracker;
		EventTracker *m_eventTracker;
		OptionTracker *m_optTracker;
		sf::RenderWindow *m_App;
		std::map<std::string, MenuItem> m_menuItems;

		sf::Shape menuBar;
		float curr_posY;
		float curr_posX;


};

}

#endif

