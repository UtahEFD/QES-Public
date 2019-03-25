/* File: Graph.h
 * Author: Matthew Overby
 *
 * TODO:
 * Handle negative values
 */

#ifndef SLUI_GRAPH_H
#define SLUI_GRAPH_H

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <fstream>
#include <map>

#include <SFML/System.hpp>

#include "Widget.h"

namespace SLUI {

struct GraphLine {

	/** @brief Default Constructor
	*/
	GraphLine();

	/** @brief Automatically sets the color of the graphline based on plotsize
	*/
	void setColor(int);

	std::vector<float> xPoints;
	std::vector<float> yPoints;
	sf::Color color;
	std::string label;
};

class Graph {

	public:
		/**  @brief Default Constructor
		*/
		Graph();

		/**  @brief Constructor
		*
		* Creates a graph object with a specified x and y positions, width 
		* and height.
		*/
		Graph(int w, int h, int px, int py);

		/**  @brief Default Destructor
		*/
		~Graph();

		/**  @brief Set the minimum and maximum values for x
		*/
		void setXMinMax( float min, float max );

		/**  @brief Set the minimum and maximum values for y
		*/
		void setYMinMax( float min, float max );

		/**  @brief Set the number of intervals for x and y
		*/
		void setInterval( int x, int y );

		/** @brief Draws the graph object
		*/
		void draw(sf::RenderWindow *g_App);

		/** @brief Copies the graph to an image file
		* @return std::string of the image's filename
		*/
		std::string printGraph(sf::RenderWindow *g_App);

		/** @brief Copies the graph to a text file
		* @return std::string of the text file's filename
		*/
		std::string printGraphData();

		/** @brief Updates the plots of the graph
		* 
		* Will remove all plots currently loaded and replace them with the
		* plots from the parameter.
		*/
		void update(std::map< std::string, GraphLine > newPlots);

		/** @brief Set the X and Y labels, respectively
		*/
		void setLabels(std::string x, std::string t);

		/** @brief Resize the graph with a new width, height, x position, and y position
		*
		* Unlike most widget classes, Graph needs to be given a new position and size
		*/
		void resizeEvent(int w, int h, int posX, int posY);

		/** @brief Changes the directory the image of the graph will be saved in
		*/
		void setSaveLocation(std::string str);

	private:
		/** @brief Calculates and creates all of the visuals for Graph
		* 
		* Creates/Positions/Resizes all visuals and stores them in lists.
		* This is only called on initial run, plot update, or window resize.
		*/
		void calcVisuals();

		/** @brief Removes the lists of visuals
		* 
		* Loops through all of the visuals and removes them, so that
		* the display lists (really just lists of sf::Shapes/Strings)
		* are empty.
		*/
		void removeVisuals();

		/** @brief Returns the position the numerical label
		* 
		* 0 = x, 1 = y
		*/
		sf::Vector2f getLabelPos( int x_or_y, sf::String num, sf::Vector2f linePos );

		sf::String m_xLabel;
		sf::String m_yLabel;
		bool plotsUpdated;
		float m_height, m_width, m_posX, m_posY;
		std::string saveLocation;

		//std::vector< GraphLine > plots;
		std::map< std::string, GraphLine > plotmap;
		std::vector< sf::Shape > m_graph;
		std::vector< sf::String > m_labels;

		float m_xMin;
		float m_xMax;
		float m_yMin;
		float m_yMax;
		int m_xInterval;
		int m_yInterval;
};

}

#endif

