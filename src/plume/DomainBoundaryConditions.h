/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
 *
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file DoaminBoundaryConditions.h 
 * @brief 
 */

#pragma once

#include <string>
#include <iostream>

class DomainBC
{
private:
  DomainBC()
  {}

protected:
  double domainStart;
  double domainEnd;

public:
  DomainBC(double dS, double dE)
  {
    domainStart = dS;
    domainEnd = dE;
  }
  ~DomainBC()
  {}
  virtual bool enforce(double &, double &) = 0;
};

class DomainBC_exiting : public DomainBC
{
public:
  DomainBC_exiting(double dS, double dE)
    : DomainBC(dS, dE)
  {}

  bool enforce(double &, double &);
};

class DomainBC_periodic : public DomainBC
{
public:
  DomainBC_periodic(double dS, double dE)
    : DomainBC(dS, dE)
  {}

  bool enforce(double &, double &);
};

class DomainBC_reflection : public DomainBC
{
public:
  DomainBC_reflection(double dS, double dE)
    : DomainBC(dS, dE)
  {}

  bool enforce(double &, double &);
};
