include_directories(${CMAKE_SOURCE_DIR})

set(UTESTS 
  ArgumentParsing_utest
  Vector3D_utest)

foreach(uexec ${UTESTS})
    add_executable(${uexec} ${BUNDLEOPTS} ${uexec}.cpp)
    target_link_libraries(${uexec} sive-util)
    target_link_libraries(${uexec} ${Boost_PROGRAM_OPTIONS_LIBRARIES})
endforeach(uexec)

ENABLE_TESTING()
INCLUDE(CTest)
foreach(uexec ${UTESTS})
  ADD_TEST(${uexec} ${CMAKE_BINARY_DIR}/utests/${uexec})
endforeach(uexec)

