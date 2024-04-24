include_directories(${PROJECT_SOURCE_DIR}/src)

IF ($CACHE{HAS_CUDA_SUPPORT})
  cuda_add_executable(util_output util_output.cpp)
  
  set(examples
          util_output)

ELSE ($CACHE{HAS_CUDA_SUPPORT})
  add_executable(util_output util_output.cpp)

  set(examples
          util_output)
      
ENDIF ($CACHE{HAS_CUDA_SUPPORT})

foreach (example ${examples})
  
  target_link_libraries(${example} qesplumecore)
  target_link_libraries(${example} qeswindscore)
  IF ($CACHE{HAS_CUDA_SUPPORT})
    target_link_libraries(${example} qeswindsgpu)
  ENDIF()
  target_link_libraries(${example} qeswindscore)
  target_link_libraries(${example} qesutil)  
  IF($CACHE{HAS_OPTIX_SUPPORT})
    target_link_libraries(${example} qesOptix)
  ENDIF()

  link_external_libraries(${example})
    
endforeach(example)
