## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
## # The following are required to uses Dart and the Cdash dashboard
##   ENABLE_TESTING()
##   INCLUDE(CTest)
set(CTEST_PROJECT_NAME "libsivelab")
set(CTEST_NIGHTLY_START_TIME "00:00:00 EST")

IF(NOT DEFINED CTEST_DROP_METHOD)
  SET(CTEST_DROP_METHOD "https")
ENDIF(NOT DEFINED CTEST_DROP_METHOD)

IF(CTEST_DROP_METHOD STREQUAL "https")
  SET(CTEST_DROP_SITE "envsim.d.umn.edu")
  SET(CTEST_DROP_LOCATION "/CDash/submit.php?project=libsivelab")
  SET(CTEST_DROP_SITE_CDASH TRUE)
  SET(CTEST_CURL_OPTIONS "CURLOPT_SSL_VERIFYPEER_OFF" "CURLOPT_SSL_VERIFYHOST_OFF")
ENDIF(CTEST_DROP_METHOD STREQUAL "https")

