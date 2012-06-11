# CMake generated Testfile for 
# Source directory: /home/alex/Desktop/repos/libsivelab
# Build directory: /home/alex/Desktop/repos/libsivelab
# 
# This file includes the relevent testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
ADD_TEST(vectorTest "/home/alex/Desktop/repos/libsivelab/tests/util_Vector3D")
SUBDIRS(cuda)
SUBDIRS(util)
SUBDIRS(network)
SUBDIRS(quicutil)
SUBDIRS(tests)
SUBDIRS(tools)
