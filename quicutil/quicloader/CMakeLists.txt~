INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/include)

add_library (sive-quicutil ${LIB_TYPE}
  # fileParser classes
  attribute.cpp attribute.h 
  attributeList.cpp attributeList.h 
  datam.cpp datam.h
  datamList.cpp datamList.h
  element.cpp element.h
  fileParser.cpp fileParser.h peekline.h
  # standard file Parser classes
  basicElements.cpp basicElements.h
  buildingArrayElement.cpp buildingArrayElement.h
  dimensionElements.cpp dimensionElements.h
  emitterArrayElement.cpp emitterArrayElement.h
  floatMatrixElement.cpp floatMatrixElement.h
  floatVectorElement.cpp floatVectorElement.h
  legacyFileParser.cpp legacyFileParser.h
  standardElements.cpp standardElements.h
  standardFileParser.cpp standardFileParser.h
  # quic loader classes
  QUICProject.h QUICProject.cpp
  QUICDataFile.h
  QUBuildings.h
  QUBuildings.cpp
  QUMetParams.h
  QUMetParams.cpp
  QUSensor.h
  QUSensor.cpp
  QUSimparams.h
  QUSimparams.cpp
  QPBuildout.h QPBuildout.cpp
  QPParams.h QPParams.cpp
  QPSource.h QPSource.cpp
  QPTurbulenceField.h QPTurbulenceField.cpp
  QUFileOptions.cpp
  QUVelocities.cpp
  #language
  language_map.h language_map.cpp
)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/quicutil)
INSTALL(TARGETS sive-quicutil LIBRARY ARCHIVE DESTINATION lib)
