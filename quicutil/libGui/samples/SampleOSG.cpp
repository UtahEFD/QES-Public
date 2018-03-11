/*	
*	Class Implementation
*
*	Before viewing this tutorial, it is recommended you understand
*	the basics explained in SampleBasic.
*/


#include "SampleOSG.h"

using namespace siveLAB;




/*	SAMPLE OPENSCENEGRAPH
*
*	This sample file gives a basic layout for rendering osg models
*	with the siveLAB GUI.
*/




//  Initalize your render window and camera in the constructor.
SampleOSG::SampleOSG(int h, int w) : Gui(h, w){

  	// Create the render window
	createRW("Sample OSG");

	// Create the camera
	vector3D initialPosition = vector3D(20.0, 20.0, 20.0);
	vector3D initialLookAtPoint = vector3D(0.0, 0.0, 0.0);
	vector3D up = vector3D(0.0, 0.0, 1.0);
	createCamera(initialPosition, initialLookAtPoint, up);
}




// Pure virtual method int Gui::display() must be declared
int SampleOSG::display(){

	loadConfig();
	m_optTracker->setActive("freemove", false);

	// Load our cow model that is in the resources directory
	osg::ref_ptr<osg::Node> loadedModel = osgDB::readNodeFile("resources/cow.osg");
	osgViewer::Viewer viewer;
	osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> gw = viewer.setUpViewerAsEmbeddedInWindow(0,0,m_width,m_height);
	osg::ref_ptr<osg::Camera> osgCam = viewer.getCamera();

	if(loadedModel){
		viewer.setSceneData(loadedModel.get());
	}
	else{
		m_eventTracker->showMessage("Model Failed to Load", 20);
	}

	viewer.realize();

   	// Start render loop
    	while (m_App->IsOpened())
    	{
		// Process events
		sf::Event Event;

		// Start control loop
		preEventLoop();
		while (m_App->GetEvent(Event))
		{
			// This function controls all menus and controls
			menuEvents(Event);
        	}
		postEventLoop();

		// Activate the render window
		setActive();

		// The osg camera is not built in to the GUI, but you can use this function to set its position and view direction
		osgCam->setViewMatrixAsLookAt( osg::Vec3d(m_camera->position.x, m_camera->position.y, m_camera->position.z ),
			osg::Vec3d(m_camera->position.x - m_camera->wVec.x, m_camera->position.y - m_camera->wVec.y, m_camera->position.z - m_camera->wVec.z),
			osg::Vec3d(m_camera->up.x, m_camera->up.y, m_camera->up.z)
		);

		viewer.frame();

		// Draw the scene and menu to the render window
		displayScene();
	}


	return EXIT_SUCCESS;
}


