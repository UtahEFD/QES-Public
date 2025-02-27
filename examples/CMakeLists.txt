include_directories(${PROJECT_SOURCE_DIR}/src)

#IF ($CACHE{HAS_CUDA_SUPPORT})
#  cuda_add_executable(util_output util_output.cpp)
  
#  set(examples
#          util_output)

#ELSE ($CACHE{HAS_CUDA_SUPPORT})
  add_executable(util_output util_output.cpp)

  set(examples
          util_output)
      
#ENDIF ($CACHE{HAS_CUDA_SUPPORT})

foreach (example ${examples})
  
  target_link_libraries(${example} PRIVATE qesplumecore)
  IF ($CACHE{HAS_CUDA_SUPPORT})
    target_link_libraries(${example} PRIVATE qeswindsgpu)
  ENDIF()
  target_link_libraries(${example} PRIVATE qeswindscore)
  target_link_libraries(${example} PRIVATE qesutil)  
  IF($CACHE{HAS_OPTIX_SUPPORT})
    target_link_libraries(${example} PRIVATE qesOptix)
  ENDIF()

  link_external_libraries(${example})
    
endforeach(example)
