/* File: GraphTracker.h
 * Author: Matthew Overby
 */

#ifndef SLUI_GRAPHTRACKER_H
#define SLUI_GRAPHTRACKER_H

#include <iostream>
#include <SFML/Graphics.hpp>
#include "OptionTracker.h"
#include "MenuItem.h"
#include "Graph.h"

namespace SLUI {

/* A GraphApp object is used when a graph is opened in a new window.
*  It will make a copy of the graph from the constructor and create
*  a new render window.
*/
struct GraphApp {

	/** @brief Constructor
	* @param A pointer to a graph object and the name of the new window
	*/
	GraphApp(Graph *g, std::string name);

	/** @brief Destructor
	*/
	~GraphApp();

	/** @brief Called by GraphTracker::checkEvent() if the window is resized
	*/
	void resizeEvent(sf::Event event);

	/** @brief Refresh the render window and redraw the items
	*/
	void redraw();

	sf::RenderWindow *g_App;
	Graph graph;
	sf::Shape menuBg;
	MenuItem *graphOptions;
};

class GraphTracker {

	public:
		/** @brief Constructor
		*/
		GraphTracker(OptionTracker*, EventTracker*);

		/** @brief Destructor
		*/
		~GraphTracker();

		/** @brief Copy all of the plots/settings of another GraphTracker
		* It does NOT copy over opened GraphApps, just all of the plots,
		* x and y labels, and save location.
		*/
		void copy( GraphTracker* );

		/** @brief Returns a pointer to the list of Graph Lines
		*/
		//std::vector< GraphLine >* getPlots();
		std::map< std::string, GraphLine >* getPlotMap();

		/** @brief Get a line from the given index of plots
		* @return a GraphLine object
		*/
		GraphLine getLine( std::string label );

		/** @brief Removes all plots from the plotlist
		*/
		void clearPlots();

		/** @brief Updates the graph
		* 
		* Loops through all of the plots and appends any changes that
		* may have been made.
		*/
		void updateGraph();

		/** @breif Add a plot to be graphed
		*
		* Add a vector of floats that can contain as many
		* items as you'd like.
		* The first vector should be your x values, the second
		* is your y, and the string is the key label.
		*/
		void addPlot( std::string label, std::vector<float> xVal, std::vector<float> yVals );

		/** @breif Launches a Graph in a new render window
		*/
		void launchWindow(Graph *graph);

		/** @breif Check for events in the launched graph render windows
		*/
		void checkEvent(sf::RenderWindow *m_App);

		/** @brief Changes the directory the image of the graph will be saved in
		*
		* If just a directory is set as the parameter, the image will be called
		* "<timestamp>.png".
		*/
		void setSaveLocation(std::string str);

		/** @brief Sets the X and Y labels of the graph
		*/
		void setLabels(std::string x, std::string y);

		void setXMinMax(float min, float max);

		void setYMinMax(float min, float max);

		bool updateWindow;

		std::string saveLocation;
		std::string xLabel, yLabel;

		float m_xMin;
		float m_xMax;
		float m_yMin;
		float m_yMax;

	private:
		std::vector< GraphApp* > g_Apps;
		//std::vector< GraphLine > plots;
		std::map< std::string, GraphLine > plotmap;
		OptionTracker *m_optTracker;
		EventTracker *m_eventTracker;
		int graphsOpened;



};

} // end namespace siveLAB

#endif
