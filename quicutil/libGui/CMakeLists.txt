cmake_minimum_required(VERSION 2.8)
project (SLUI)

find_package(OpenGL REQUIRED)

IF(APPLE)
  MESSAGE(STATUS "Using local dir or SFML 1.6")
  SET(SFMLDIR ${CMAKE_SOURCE_DIR}/Externals/SFML-1.6)
ELSE(APPLE)
  SET(SFMLDIR /usr)
ENDIF(APPLE)

find_package(SFML 1.6 COMPONENTS system window graphics audio REQUIRED)

find_package(Boost REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(Boost COMPONENTS thread REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)

include_directories(include)
include_directories(src)
include_directories(samples)
include_directories(bin)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${GLEW_INCLUDE_PATH})
include_directories(${Boost_INCLUDE_DIR})

include_directories(${SFML_INCLUDE_DIR})
link_libraries(${SFML_LIBRARIES})

link_libraries(${OPENGL_LIBRARY})

set( BOOST_LINK_LIBS ${Boost_PROGRAM_OPTIONS_LIBRARIES} ${Boost_THREAD_LIBRARY}
	${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} )
link_libraries(${BOOST_LINK_LIBS})

add_library (Gui
	src/Gui.cpp		include/Gui.h
	src/Widget.cpp		include/Widget.h
	src/Camera.cpp 		include/Camera.h
	src/Calendar.cpp 	include/Calendar.h
	src/Button.cpp 		include/Button.h
	src/vector3D.cpp 	include/vector3D.h
	src/vector4D.cpp 	include/vector4D.h
	src/Keys.cpp 		include/Keys.h
	src/Matrix3.cpp 	include/Matrix3.h
	src/Matrix4.cpp 	include/Matrix4.h
	src/MenuItem.cpp 	include/MenuItem.h
	src/DateWindow.cpp 	include/DateWindow.h
	src/PrefWindow.cpp	include/PrefWindow.h
	src/TextWindow.cpp 	include/TextWindow.h
	src/GraphWindow.cpp 	include/GraphWindow.h
	src/ClickcellWindow.cpp include/ClickcellWindow.h
	src/TextBox.cpp 	include/TextBox.h
	src/Keys.cpp 		include/Keys.h
	src/Option.cpp 		include/Option.h
	src/Console.cpp 	include/Console.h
	src/Config.cpp 		include/Config.h
	src/GuiStats.cpp 	include/GuiStats.h
	src/LoadScreen.cpp 	include/LoadScreen.h
	src/Legend.cpp 		include/Legend.h
	src/Graph.cpp 		include/Graph.h
	src/Window.cpp		include/Window.h
	src/WindowManager.cpp 	include/WindowManager.h
	src/OptionTracker.cpp 	include/OptionTracker.h
	src/GraphTracker.cpp 	include/GraphTracker.h
	src/EventTracker.cpp 	include/EventTracker.h
	src/MenuTracker.cpp 	include/MenuTracker.h
	src/CellTracker.cpp	include/CellTracker.h
	src/Scrollbar.cpp 	include/Scrollbar.h
	src/VolumeCell.cpp 	include/VolumeCell.h
	src/FileFinder.cpp 	include/FileFinder.h
	src/ColorScale.cpp 	include/ColorScale.h
	src/FisheyeWidget.cpp 	include/FisheyeWidget.h
	samples/SampleBasic.cpp samples/SampleBasic.h
	samples/SampleGraph.cpp samples/SampleGraph.h
	samples/SampleClickworld.cpp samples/SampleClickworld.h
#	samples/SampleOSG.cpp	samples/SampleOSG.h
)

#add_library (Tui
#	src/Tui.cpp		include/Tui.h
#	src/OptionTracker.cpp 	include/OptionTracker.h
#)


#add_executable (bin/basic samples/basic.cpp)
#add_executable (bin/colorbartest samples/colorbartest.cpp)

#add_executable (bin/graph samples/graph.cpp)
#target_link_libraries (bin/graph Gui ${BOOST_LINK_LIBS})

# add_executable (clickworld samples/clickworld.cpp)
#add_executable (bin/osg samples/osg.cpp)
#target_link_libraries (bin/basic Gui ${BOOST_LINK_LIBS})
#target_link_libraries (bin/colorbartest Gui ${BOOST_LINK_LIBS})
#target_link_libraries(clickworld Gui ${BOOST_LINK_LIBS})

#target_link_libraries(bin/osg Gui)
#target_link_libraries(bin/osg ${LIBS})


