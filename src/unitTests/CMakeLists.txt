include_directories(${CMAKE_SOURCE_DIR})

add_library(unitTestLib ${LIB_TYPE} 
	    test_DTEHeightField.h test_DTEHeightField.cpp
            test_WINDSGeneralData.cpp
            test_TURBGeneralData.cpp
	    test_Turbulence.cpp
            util.h util.cpp)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/unitTests)
