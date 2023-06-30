include_directories(${PROJECT_SOURCE_DIR}/src)

IF ($CACHE{HAS_CUDA_SUPPORT})

    CUDA_ADD_EXECUTABLE(plume_sinusoidal
            ../unitTests/test_WINDSGeneralData.h ../unitTests/test_WINDSGeneralData.cpp
            plume_sinusoidal.cpp)

    CUDA_ADD_EXECUTABLE(plume_uniform
            ../unitTests/test_WINDSGeneralData.h ../unitTests/test_WINDSGeneralData.cpp
            plume_uniform.cpp)

ELSE ($CACHE{HAS_CUDA_SUPPORT})

    add_executable(plume_sinusoidal
            ../unitTests/test_WINDSGeneralData.h ../unitTests/test_WINDSGeneralData.cpp
            plume_sinusoidal.cpp)

    add_executable(plume_uniform
            ../unitTests/test_WINDSGeneralData.h ../unitTests/test_WINDSGeneralData.cpp
            plume_uniform.cpp)

ENDIF ($CACHE{HAS_CUDA_SUPPORT})



set(REGRESSTESTS
  plume_sinusoidal
  plume_uniform)

foreach (regresstest ${REGRESSTESTS})

  target_link_libraries(${regresstest} qesplumecore)
  target_link_libraries(${regresstest} qeswindscore)
  IF ($CACHE{HAS_CUDA_SUPPORT})
    target_link_libraries(${regresstest} qeswindsgpu)
  ENDIF()
  target_link_libraries(${regresstest} qeswindscore)
  target_link_libraries(${regresstest} qesutil)
  IF($CACHE{HAS_OPTIX_SUPPORT})
    target_link_libraries(${regresstest} qesOptix)
  ENDIF()
  link_external_libraries(${regresstest})
  target_link_libraries(${regresstest} Catch2::Catch2WithMain)
  
  if (ENABLE_REGRESSION_TESTS)
    add_test(regression_test::${regresstest} ${regresstest})
  endif (ENABLE_REGRESSION_TESTS)
  
endforeach(regresstest)

