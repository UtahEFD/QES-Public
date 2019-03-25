/* File: MenuItem.h
 * Author: Matthew Overby
 *
 * TODO:
 * Make this class more readable/easy to understand.
 * This is one of those classes that has been altered so
 * many times to fit specific needs, it's become nosensical
 * and very difficult to use.
 */

#ifndef SLUI_MENUITEM_H
#define SLUI_MENUITEM_H

#include "Widget.h"
#include "WindowManager.h"
#include "EventTracker.h"
#include "OptionTracker.h"

namespace SLUI {

namespace MenuType {
	const int Main = 1;
	const int Window = 2;
	const int Index = 3;
	const int Hidden = 4;
	const int Func = 5;
	const int List = 6;
}

/** @brief SubItems are items within a MenuItem
*
* When a MenuItem is clicked (dropped) the SubItems become viewable.
* There are four types of SubItems:
*	- Window: when clicked, will open/close a window
*	- Index: when clicked, will return a constant integer value
*	- Func: when clicked, calls the associated function
*	- List: when clicked, sets the list option to that value
*/
class SubItem : public Widget {

	public:
		/** @brief Default Constructor
		*/
		SubItem();

		/** @brief Constructor
		*/
		SubItem(int posX, int posY, std::string lab, sf::RenderWindow *app);

		/** @brief Draws the submenu item
		*/
		void draw();

		/** @brief Moves the submenu item
		*/
		void move(float newX, float newY);

		/** @brief Highlights the submenu item is the mouse is over it
		*/
		void highlight();

		int type;
		int window;
		int index;
		std::string command;
		sf::String label;
};

class MenuItem : public Widget {

	public:
		static const int DefaultHeight = 30;
		static const int DefaultWidth = 160;
		static const int DefaultLabelSize = 18;

		/** @brief Default Constructor
		*/
		MenuItem();

		/** @brief Constructor
		*
		* Creates a new main menu item (top of render window)
		*/
		MenuItem(int posX, int posY, std::string lab, EventTracker *evt, OptionTracker *opt, WindowManager *wnt, sf::RenderWindow *app);

		/** @brief Constructor
		*
		* Creates a mouse menu
		*/
		MenuItem(EventTracker *evt, sf::RenderWindow *app);

		/** @brief Draws the menu item and if dropped, its submenu items
		*/
		void draw();

		/** @brief Adds a window submenu item to its list
		*/
		void addWindowItem(unsigned int val, std::string s);

		/** @brief Adds an index submenu item to its list
		*/
		void addIndexItem(int val, std::string s);

		/** @brief Adds a list option item to its list
		*/
		void addListItem(std::string command, std::string label);

		/** @brief Adds a function submenu item to its list
		* When pressed, the function associated with the menu item
		* (from m_eventTracker->m_eventFuncs) will execute
		*/
		void addFuncItem(std::string command, std::string label);

		/** @brief Checks to see if the mouse is over and highlight if needed
		*/
		void highlight();

		/** @brief Called in the event loop, checks to see if the event applies
		*/
		void checkEvent();

		/** @brief Returns the index value of an index submenu if it was clicked
		* @return int value of the item selected
		*/
		int getIndexClicked();

		/** @brief Returns the label a function submenu if it was clicked
		* @return string value of the label of the function
		*/
		std::string getFuncClicked();

		/** @brief Checks to see what the mouse is over and handle the click
		*/
		void mouseClicked();

		/** @brief Checks to see what the mouse is over and handle the click
		*
		* This is used by the cellTracker's right click menu
		*/
		void mouseClickedRight();

		/** @brief Toggles the dropped bool value and color of the MenuItem
		*/
		void switchDropped();

		/** @brief Set the value of drop and change to the appropriate highlight color
		*/
		void setDropped(bool drop);

		/** @brief Sets the label to the new string
		*/
		void setLabel(std::string newLabel);

		/** @brief Sets the label at index of a SubItem
		*/
		void setSubItemLabel(int index, std::string newLabel);

		/** @brief Called by the mouse menu to reset it's position to the mouse pointer
		*/
		void move(float newX, float newY);


		bool dropped;
		int type;

	private:
		/** @brief Stretches the background width to fit the parameter string
		* 
		* Does not change the position of the item, just m_width.  It also does not
		* change any shapes that may correspond with the widget.
		*/
		virtual void fitString(sf::String str);

		sf::String label;
		EventTracker *m_eventTracker;
		WindowManager *m_winManager;
		OptionTracker *m_optTracker;
		sf::Vector2f dropIndex;
		std::map< int, SubItem > m_items;


};

}

#endif
