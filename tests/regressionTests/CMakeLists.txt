include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(.)

IF ($CACHE{HAS_CUDA_SUPPORT})

    add_executable(plume_sinusoidal
            ../unitTests/test_WINDSGeneralData.h ../unitTests/test_WINDSGeneralData.cpp
            plume_sinusoidal.cpp)

ELSE ($CACHE{HAS_CUDA_SUPPORT})

    add_executable(plume_sinusoidal
            ../unitTests/test_WINDSGeneralData.h ../unitTests/test_WINDSGeneralData.cpp
            plume_sinusoidal.cpp)

ENDIF ($CACHE{HAS_CUDA_SUPPORT})



set(REGRESSTESTS
        plume_sinusoidal)

foreach (regresstest ${REGRESSTESTS})
    target_link_libraries(${regresstest} qesplumecore)

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

    if (ENABLE_UNITTESTS)
        add_test(regression_test-${regresstest} ${regresstest})
    endif (ENABLE_UNITTESTS)
endforeach(regresstest)

