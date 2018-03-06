/*
 *  handleGraphicsArgs.cpp
 *
 *  Created by Pete Willemsen on 10/6/09.
 *  Copyright 2009 Department of Computer Science, University of Minnesota-Duluth. All rights reserved.
 *
 * This file is part of libSIVELab (libsivelab).
 *
 * libsivelab is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libsivelab is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with libsivelab.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "handleGraphicsArgs.h"

using namespace sivelab;

GraphicsArgs::GraphicsArgs()
  : verbose(false), width(100), height(100), 
    aspectRatio(1.0), useShadow(true), bgColor(0.0, 0.0, 0.0),
    numCpus(1), rpp(1), inputFileName(""), outputFileName("")
{
}

void GraphicsArgs::process(int argc, char *argv[])
{
  ArgumentParsing argParser;

  argParser.reg("help", "help/usage information", ArgumentParsing::NONE, '?');
  argParser.reg("verbose", "turn on verbose output", ArgumentParsing::NONE, 'v');
  argParser.reg("inputfile", "input file name to use", ArgumentParsing::STRING, 'i');
  argParser.reg("outputfile", "output file name to use", ArgumentParsing::STRING, 'o');
  argParser.reg("numcpus", "num of cores to use", ArgumentParsing::INT, 'n');
  argParser.reg("width", "width of image (default is 100)", ArgumentParsing::INT, 'w');
  argParser.reg("height", "height of image (default is 100)", ArgumentParsing::INT, 'h');
  argParser.reg("aspect", "aspect ratio in width/height of image (default is 1)", ArgumentParsing::FLOAT, 'a');
  argParser.reg("rpp", "rays per pixel (default is 1)", ArgumentParsing::INT, 'r');

  argParser.processCommandLineArgs(argc, argv);

  if (argParser.isSet("help"))
    {
      argParser.printUsage();
      exit(EXIT_SUCCESS);
    }

  verbose = argParser.isSet("verbose");
  if (verbose) std::cout << "Verbose Output: ON" << std::endl;
  
  argParser.isSet("width", width);
  if (verbose) std::cout << "Setting width to " << width << std::endl;
  
  argParser.isSet("height", height);
  if (verbose) std::cout << "Setting height to " << height << std::endl;
  
  argParser.isSet("aspect", aspectRatio);
  if (verbose) std::cout << "Setting aspect ratio to " << aspectRatio << std::endl;

  argParser.isSet("numcpus", numCpus);
  if (verbose) std::cout << "Setting num cpus to " << numCpus << std::endl;

  argParser.isSet("rpp", rpp);
  if (verbose) std::cout << "Setting rays per pixel to " << rpp << std::endl;
  
  argParser.isSet("inputfile", inputFileName);
  if (verbose) std::cout << "Setting inputFileName to " << inputFileName << std::endl;

  argParser.isSet("outputfile", outputFileName);
  if (verbose) std::cout << "Setting outputFileName to " << outputFileName << std::endl;
}

