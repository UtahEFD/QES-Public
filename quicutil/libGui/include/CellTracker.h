/* File: CellTracker.h
 * Author: Matthew Overby
 *
 * TODO:
 * This class needs a better cell type structure.
 * Right now the differing calls between patches and
 * volumes is dirty and not very functional.
 */


#ifndef SLUI_CELLTRACKER_H
#define SLUI_CELLTRACKER_H

#include "Widget.h"
#include "Button.h"
#include "vector3D.h"
#include "EventTracker.h"
#include "MenuItem.h"
#include "VolumeCell.h"

namespace SLUI {

struct clickRay {

	clickRay();
	vector3D origin;
	vector3D direction;
	static const float tmin;
	static const float tmax;
	float closest;
	int hitID;
};

namespace CellType {

	const int Patch = 1;

	static const int DOWN_FACE = 0;
	static const int UP_FACE = 1;
	static const int NORTH_FACE = 2;
	static const int EAST_FACE = 3;
	static const int SOUTH_FACE = 4;
	static const int WEST_FACE = 5;
}

class Cell {

	public:
		/** @brief Default Constructor
		*/
		Cell();

		/** @brief Check for a ray/patch intersection
		* Upon intersection, report t value and id to ray
		*/
		void intersect( clickRay *ray );

		bool selected;
		unsigned int id;
		int type, face;
		std::vector<std::string> stats;

		vector3D renderColor, highlightColor;
		vector3D a, b, c, d;

		vector3D normal;
		vector3D anchor;
		vector3D v1;
		vector3D v2;

	private:
};



class CellTracker {

	public:
		/** @brief Default Constructor
		*/
		CellTracker(EventTracker* evT, sf::RenderWindow* app);

		/** @brief Default Destructor
		*/
		~CellTracker();

		/** @brief Add a renderable volume to the list
		*/
		void addVolume( unsigned int id, VolumeCell vc );

		/** @brief Adds a patch to the cell list
		* ID will be added to the selected list if clicked
		*/
		void addPatch( unsigned int id, int face, vector3D anchor, vector3D v1, vector3D v2 );

		/** @brief Changes a cell's render color
		*/
		void setCellColor( unsigned int id, vector3D newColor );

		/** @brief Changes a cell's render color
		*/
		void setCellColor( unsigned int id, vector4D newColor );

		/** @brief Changes a cell's highlight (selected) color
		*/
		void setCellHighlightColor( unsigned int id, vector3D newColor );

		/** @brief Add a stat to a patch which is viewable in the clickcell window
		*/
		void addStat( unsigned int id, std::string stat );
		
		/** @brief clear all stats for a patch which is viewable in the clickcell window
		*/
		void clearStat( unsigned int id);

		/** @brief Return a string containing all of the stats for a cell
		* @return a string
		*/
		std::string getStats( unsigned int id );

		/** @brief Draws all renderable boxes that were added using addBox
		*/
		//void drawBoxes();

		/** @brief Draws all renderable trees that were added using addVolume
		*/
		void drawTrees();

		/** @brief Draws only a certain "layer" of aircells
		* 0   : Draw nothing
		* 1-X : One layer of volumes increasing in the Z direction
		*/
		void drawAircells( int layer );

		/** @brief Draw only cells that are selected (highlighted)
		*/
		void drawSelected();

		/** @brief Finds the object clicked on (shift-click)
		* 
		* If worldClick is active, it finds the object clicked on
		* and sets lastCellSelected to its id.  It also changes
		* that cube's selected variable to true.
		*/
		void checkIntersect();
		void checkIntersect( clickRay &ray );

		/** @brief Check the most recent event and see if it applies
		*/
		void checkEvent();

		/** @brief Return a copy of the list of selected cells (by id)
		*/
		std::vector<unsigned int> getSelected(){ return selectedCells; }

		/** @brief Clear the whole cell list
		*/
		void clearCellList();

		/** @brief Clear the list of selected cells
		*/
		void clearSelected();

		/** @brief Set a cell as selected
		*/
		void setSelected( unsigned int id );

		/** @brief Remove a specific cell from the cell list
		*/
		void removeCell( unsigned int id );

		/** @brief Remove a specific cell from the selected list
		*/
		void removeSelected( unsigned int id );

		/** @brief Test function to print cell stuff
		*/
		void testPrintCells();

		/** @brief Get the size of the cell list
		*/
		unsigned int size(){ return cellList.size(); }

		/** @brief Get the size of the volumeList
		*/
		unsigned int volumes(){ return volumeList.size(); }

		void setVolumeRenderColor( unsigned int id, vector3D color );

		/** @brief 
		*/
		void renderCells();

		/** @brief 
		*/
		bool addRenderableCell( unsigned int id, RenderableCell *cell );

		/** @brief 
		*/
		RenderableCell* getRenderableCell( unsigned int id );

	private:
		std::map<unsigned int, Cell> cellList;
		std::map<unsigned int, VolumeCell> volumeList;

		RenderMap m_renderCells;
		std::map<unsigned int, DataCell*> m_dataCells;

		std::vector<unsigned int> selectedCells;
		sf::RenderWindow* m_App;
		EventTracker* m_eventTracker;

};

}

#endif

