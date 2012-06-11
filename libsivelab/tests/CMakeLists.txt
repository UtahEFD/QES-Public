include_directories(${CMAKE_SOURCE_DIR})
link_directories(${CMAKE_CURRENT_BUILD_DIR}/network)

# SET(BUNDLEOPTS MACOSX_BUNDLE)

set(NETTESTS network_IPConversions network_TCPServer network_TCPClient network_UDPSender network_UDPReceiver network_UDPDataSender network_UDPDataReceiver)

foreach(netexec ${NETTESTS})
  add_executable(${netexec} ${netexec}.cpp)

  target_link_libraries(${netexec} sive-network)
  target_link_libraries(${netexec} sive-util)
  target_link_libraries(${netexec} ${Boost_PROGRAM_OPTIONS_LIBRARIES})
endforeach(netexec)

set(UTILTESTS util_logstream util_random util_Vector3D util_Timer)
foreach(utilexec ${UTILTESTS})
    add_executable(${utilexec} ${BUNDLEOPTS} ${utilexec}.cpp)

    target_link_libraries(${utilexec} sive-util)
endforeach(utilexec)

add_executable(util_pngWrite util_pngWrite.cpp)
target_link_libraries(util_pngWrite sive-util)
target_link_libraries(util_pngWrite ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(util_pngWrite ${PNG_LIBRARIES})

add_executable(quicLoaderTest quicLoaderTest.cpp)
target_link_libraries(quicLoaderTest sive-quicutil)
target_link_libraries(quicLoaderTest sive-util)
target_link_libraries(quicLoaderTest ${Boost_PROGRAM_OPTIONS_LIBRARIES})

INSTALL(TARGETS ${NETTESTS} DESTINATION bin)
INSTALL(TARGETS ${UTILTESTS} DESTINATION bin)
INSTALL(TARGETS util_pngWrite DESTINATION bin)
INSTALL(TARGETS quicLoaderTest DESTINATION bin)
