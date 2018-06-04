include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${CMAKE_BINARY_DIR}/src/util)

set(BASETESTS argparser)
foreach(basetest ${BASETESTS})
    add_executable(${basetest} ${basetest}.cpp)
    target_link_libraries(${basetest} qesutil)
    target_link_libraries(${basetest} ${Boost_PROGRAM_OPTIONS_LIBRARIES})
endforeach(basetest)

