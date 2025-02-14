include_directories(${PROJECT_SOURCE_DIR}/src)

unset(REGRESSTESTS)

add_executable(plume_sinusoidal plume_sinusoidal.cpp)
list(APPEND REGRESSTESTS plume_sinusoidal)

add_executable(plume_uniform plume_uniform.cpp)
list(APPEND REGRESSTESTS plume_uniform)

foreach (regresstest ${REGRESSTESTS})
  
  target_link_libraries(${regresstest} PRIVATE qesplumecore)
  IF ($CACHE{HAS_CUDA_SUPPORT})
    target_link_libraries(${regresstest} PRIVATE qeswindsgpu)
  ENDIF()
  target_link_libraries(${regresstest} PRIVATE qeswindscore)
  target_link_libraries(${regresstest} PRIVATE qesutil)
  IF($CACHE{HAS_OPTIX_SUPPORT})
    target_link_libraries(${regresstest} PRIVATE qesOptix)
  ENDIF()
  link_external_libraries(${regresstest})
  target_link_libraries(${regresstest} PRIVATE Catch2::Catch2WithMain)
  
  add_test(NAME regression_test::${regresstest} COMMAND ${regresstest})
  
endforeach(regresstest)

