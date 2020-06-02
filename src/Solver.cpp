/*
 *
 * CUDA-URB
 * Copyright (c) 2019 Behnam Bozorgmehr
 * Copyright (c) 2019 Jeremy Gibbs
 * Copyright (c) 2019 Eric Pardyjak
 * Copyright (c) 2019 Zachary Patterson
 * Copyright (c) 2019 Rob Stoll
 * Copyright (c) 2019 Pete Willemsen
 *
 * This file is part of CUDA-URB package
 *
 * MIT License
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include "Solver.h"

using std::cerr;
using std::endl;
using std::vector;
using std::cout;

// duplication of this macro
#define CELL(i,j,k,w) ((i) + (j) * (nx+(w)) + (k) * (nx+(w)) * (ny+(w)))

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

/**< This function is showing progress of the solving process by printing the percentage */

void Solver::printProgress (float percentage)
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}


/**< \fn Solver
* This function is assigning values read by URBImputData to variables
* used in the solvers - this is only meant work with CUDA-URB!
 */

Solver::Solver(const URBInputData* UID, URBGeneralData * UGD)
    : alpha1 (1),
      alpha2 (1),
      eta( pow((alpha1/alpha2), 2.0) ),
      A( pow( (UGD->dx/UGD->dy), 2.0 ) ),
      B( eta*pow( (UGD->dx/UGD->dz), 2.0) ),
      itermax( UID->simParams->maxIterations )

{
  tol = UID->simParams->tolerance;
}
