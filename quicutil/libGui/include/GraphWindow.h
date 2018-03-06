/* File: GraphWindow.h
 * Author: Matthew Overby
 *
 * NOTE:
 * This class isn't meant to be used for your own purposes.
 * If you look at the implementation file you'll see references
 * to options that were never initialized in Gui.cpp.
 * You should really only use it as a reference on how to
 * interact with and use graphs.
 */

#ifndef SLUI_GRAPHWINDOW_H
#define SLUI_GRAPHWINDOW_H

#include "Button.h"
#include "Widget.h"
#include "TextBox.h"
#include "Graph.h"
#include "Window.h"
#include "OptionTracker.h"
#include "EventTracker.h"
#include "GraphTracker.h"

namespace SLUI {

class GraphWindow : public Window {

	public:
		/** @brief Constructor
		*/
		GraphWindow(int newid, WindowController *wnC, GraphTracker *grT, 
			OptionTracker *opT, EventTracker *evt, sf::RenderWindow* app);

		/** @brief Default Destructor
		*/
		~GraphWindow();

		/** @brief Draws the graph window's elements
		*/
		void draw();

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

		/** @brief Refreshes the graph visuals and applies newly changed options
		*/
		void update();

	private:
		/** @brief Checks to see what the mouse is over and highlight buttons if needed
		* 
		* This function loops through all of the buttons, calling their highlight function.
		*/
		void highlight();

		/** @brief Loops through all of the buttons and executes their action
		*/
		void mouseClicked();

		Button *closeButton;
		Button *exportButton;
		Button *redrawButton;
		Button *newWinButton;
		Button *settingsButton;
		Button *clearPlotsButton;
		Button *printButton;
		Button *showButton;

		OptionTracker *m_optTracker;
		GraphTracker *m_graphTracker;

		Graph *m_graph;

};

}

#endif

