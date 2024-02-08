/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
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
#include "util/QEStime.h"

/**
 * @class WINDSOutputVisualization
 * @brief Specialized output classes derived from QESNetCDFOutput form
 * cell center data (used primarily for visualization)
 */
class WINDSOutputVisualization : public QESNetCDFOutput
{
private:
  WINDSOutputVisualization() {}

public:
  WINDSOutputVisualization(WINDSGeneralData *, WINDSInputData *, const std::string &);
  ~WINDSOutputVisualization()
  {}

  /**
   * :document this:
   */
  void save(QEStime);

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
  std::vector<float> u_out, v_out, w_out, mag_out;
  ///@}

  WINDSGeneralData *m_WGD; /**< :document this: */
};
