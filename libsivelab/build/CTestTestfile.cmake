# CMake generated Testfile for 
# Source directory: /home/alex/Desktop/project/libsivelab
# Build directory: /home/alex/Desktop/project/libsivelab/build
# 
# This file includes the relevent testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
ADD_TEST(vectorTest "/home/alex/Desktop/project/libsivelab/build/tests/util_Vector3D")
SUBDIRS(cuda)
SUBDIRS(util)
SUBDIRS(network)
SUBDIRS(quicutil)
SUBDIRS(tests)
SUBDIRS(tools)
