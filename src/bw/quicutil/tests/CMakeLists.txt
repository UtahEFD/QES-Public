include_directories(${CMAKE_SOURCE_DIR})

set(UTILTESTS util_logstream util_random util_Vector3D util_Timer)
foreach(utilexec ${UTILTESTS})
    add_executable(${utilexec} ${BUNDLEOPTS} ${utilexec}.cpp)

    target_link_libraries(${utilexec} util)
endforeach(utilexec)

add_executable(quicLoaderTest quicLoaderTest.cpp)
target_link_libraries(quicLoaderTest quicloader)
target_link_libraries(quicLoaderTest util)
target_link_libraries(quicLoaderTest ${Boost_PROGRAM_OPTIONS_LIBRARIES})

add_executable(quicQueryBuildings quicQueryBuildings.cpp)
target_link_libraries(quicQueryBuildings quicloader)
target_link_libraries(quicQueryBuildings util)
target_link_libraries(quicQueryBuildings ${Boost_PROGRAM_OPTIONS_LIBRARIES})

INSTALL(TARGETS ${UTILTESTS} DESTINATION bin)
INSTALL(TARGETS quicLoaderTest DESTINATION bin)
