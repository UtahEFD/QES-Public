/* File: Gui.h
 * Author: Matthew Overby
 */

#ifndef SLUI_GUI_H
#define SLUI_GUI_H

#include "Camera.h"
#include "MenuItem.h"
#include "PrefWindow.h"
#include "TextWindow.h"
#include "GraphWindow.h"
#include "DateWindow.h"
#include "ClickcellWindow.h"
#include "Config.h"
#include "GuiStats.h"
#include "LoadScreen.h"
#include "Legend.h"
#include "WindowManager.h"
#include "EventTracker.h"
#include "MenuTracker.h"
#include "FileFinder.h"

namespace SLUI {

/** @brief See sample/SampleGui for instructions on how to implement the Gui
*
* Functions ending in (required) MUST be called at certain points in your
* derived Gui.  See sample/SampleGui for more information.
*/
class Gui {

	public:
		/** @brief Constructor
		* 
		* Initializes menu to null, and creates
		* a new SFML render window object.
		*/
		Gui();

		/** @brief Constructor
		* Creates a GUI object with width and height
		*/
		Gui(int width, int height);

		/** @brief Virtual Destructor
		*/
		virtual ~Gui();

		/** @brief Does the specified function on specifed key press
		* 
		* This method should only be used in the events loop (while loop inside
		* main display loop).  If the specified key (desiredKey) is pressed,
		* it will execute the function from the pointer, and display a load
		* screen while executing the function.
		* The function must return void and have no parameters.  A typical call
		* in the event loop will look like this:
		* addKeyFunction( Event, m_optTracker->getString("someButton"), &DerivedGui::someFunction );
		*/
		Keys GuiKey;
		template <typename T> void addKeyFunction(sf::Event Event, std::string desiredKey, void (T::*f)()){
			if (Event.Type == sf::Event::KeyPressed && Event.Key.Code == GuiKey.findKey(desiredKey) &&
			    !m_winManager->isOpen()){

				preSFML();
				loadScreen->draw();
				m_App->Display();
				postSFML();
				(((T*)this)->*f)();
			}
		}

	protected:

		/** @brief Make the world clickable (must be called in derived constructor!)
		*
		* If you are rendering the world with Cubes from ClickCell.h,
		* you have the option of using shift-click to display statistics
		* for each cell.  enable world click (in the derived
		* constructor) if this is the case.  Make sure you push back your
		* cubes into cubeList.
		*/
		void enableWorldClick(bool val);

		/** @brief Allows the graphing features to be used (must be called in derived constructor!)
		*
		* This function will create all of the options, windows, and menus for graphing.
		*/
		void enableGraphing(bool val);

		/** @brief Set whether to hide gui visuals or not.
		*
		* This can also be toggled by pressing F12.
		*/
		void setHideMenu(bool val);

		/** @brief Set the perspective far plane distance
		*/
		void setFarPlane(float far);

		/** @brief Displays the SFML menu items and calls m_App->Display() (required)
		* 
		* Disables wireframe mode and then displays (if open)
		* preferences window, menu, stats, console. Then it re-enables
		* wireframe (if active) before m_App->Display() is called.
		*/
		void displayMenu();

		/** @brief Loads settings from the configuration file (required)
		* 
		* This should be called at the end of your derived gui constructor!
		*/
		void loadConfig();

		/** @brief creates the gui's camera (required)
		* 
		* Takes in the position, look at point, and up vectors and creates
		* a new camera.
		*/
		void createCamera(vector3D, vector3D, vector3D);

		/** @brief Creates a new render window with new title
		*/
		void createRW(std::string);

		/** @brief Adds a new bool option
		* 
		* (To see more information on bool options, see Option.h)
		* Adds a new option to Options map table with specified key (command)
		* and initial bool value (active).
		* It also adds a radial button to the preferences window with the
		* specified label (label).  The last param is to associate to which
		* menu it will appear.
		*/
		void addBoolOption(std::string, std::string, bool, unsigned int);

		/** @brief Adds a new value option
		* 
		* (To see more information on value options, see Option.h)
		* Adds a new option to Options map table with specified key (command)
		* and initial value (initVal).
		*/
		void addValueOption(std::string command, std::string label, 
			float initVal, unsigned int window);

		/** @brief Adds a new value option with a specified upper bound and lower bound
		* 
		* (To see more information on value options, see Option.h)
		* Adds a new option to Options map table with specified key (command)
		* and initial value (initVal).
		*/
		void addValueOption(std::string command, std::string label, 
			float initVal, float minVal, float maxVal, unsigned int window);

		/** @brief Adds a new key option
		* 
		* For more information on key options, see Keys.h
		* Creates a new key option in Keys map table with specified key (command)
		* and keyboard key (desiredKey).
		* Also adds a key button to preferences menu with specified label (label).
		* To access the key value as an std::string, use "config.Keys[<command>].toString"
		*/
		void addKeyOption(std::string, std::string, std::string, unsigned int);

		/** @brief Adds a new list option
		* 
		* (To see more information on value options, see Option.h)
		* Adds a new option to Options map table with specified key (command)
		* and selectable options (opts).
		*/
		void addListOption(std::string command, std::string label, std::vector<std::string> opts, unsigned int window);

		/** @brief Must be called before the event loop (required)
		*/
		void preEventLoop();

		/** @brief Must be called after the event loop (required)
		*/
		void postEventLoop();

		/** @brief Controls the menu events (required)
		* 
		* This function must be called in the events while loop (the while loop
		* inside main display loop).
		* This function controls nearly all keyboard and mouse actions related to
		* the menu, preferences window, console, clicking the world, etc...
		*/
		void menuEvents(sf::Event);

		/** @brief Disables wireframe mode if active
		* 
		* This function disables opengl wireframe mode,
		* but does not toggle the wireframe option.
		* Use it before displaying sfml menu item.
		*/
		void preSFML();

		/** @brief Enables wireframe mode if active
		* 
		* This function enables opengl wireframe mode,
		* but does not toggle the wireframe option.
		* Use it after displaying sfml menu item.
		*/
		void postSFML();

		/** @brief Sets the RenderWindow to active (required)
		* 
		* This function sets m_App to active and must be called
		* in derived render loop.  It also calls m_camera->setGLView()
		* and clearGL().
		*/
		void setActive();

		/** @brief Pure virtual function display (required)
		*/
		virtual int display() = 0;

		int m_height, m_width;

		sf::RenderWindow *m_App;
		Camera *m_camera;
		LoadScreen *loadScreen;
		Legend *m_legend;
		Config *m_config;
		GuiStats *m_stats;

		OptionTracker *m_optTracker;
		GraphTracker *m_graphTracker;
		WindowManager *m_winManager;
		WindowController *m_winController;
		EventTracker *m_eventTracker;
		MenuTracker *m_menuTracker;
		CellTracker *m_cellTracker;


		float bgColor[4];

	private:

		/** @brief Clears openGL values
		* 
		* This method must be called in the display loop before a scene is drawn
		* It clears the buffers/etc...
		*/
		void clearGL();

		/** @brief Initializes memory for Gui's pointers.
		* 
		* Do not call this function, it is called by the constructor
		*/
		void initializeMemory();

		/** @brief Initializes the gui
		* 
		* Do not call this function, it is called in createRW().
		* -Creates a new preference window, stats window, console, and menu
		* -Adds default buttons/options
		* -Loads user's config file into options
		*/
		void initializeMenu();

		bool worldClick;
		bool graphing;
		float moveAccel;
		
};

}

#endif

