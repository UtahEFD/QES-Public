/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file WINDSOutputVisualization.h */

#pragma once

#include <string>
#include <vector>
#include <algorithm>

#include "WINDSGeneralData.h"
#include "WINDSInputData.h"
#include "util/QESNetCDFOutput.h"

/**
 * @class WINDSOutputVisualization
 * @brief Specialized output classes derived from QESNetCDFOutput form
 * cell center data (used primarily for visualization)
 */
class WINDSOutputVisualization : public QESNetCDFOutput
{
public:
  WINDSOutputVisualization()
    : QESNetCDFOutput()
  {}
  WINDSOutputVisualization(WINDSGeneralData *, WINDSInputData *, std::string);
  ~WINDSOutputVisualization()
  {}

  /**
   * :document this:
   */
  void save(ptime);

protected:
  /**
   * :document this:
   */
  void setAllOutputFields();

private:
  ///@{
  /** :document this: */
  std::vector<float> x_out, y_out, z_out;
  ///@}

  ///@{
  /**< :document this: */
  std::vector<int> icellflag_out, icellflag2_out;
  ///@}

  ///@{
  /** :document this: */
  std::vector<double> u_out, v_out, w_out, mag_out;
  ///@}

  WINDSGeneralData *WGD_; /**< :document this: */
};
