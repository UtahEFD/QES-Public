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

/**
 * @file QESFileOutput_v2.cpp
 */
#include "QESFileOutput_v2.h"
#include "DataSource.h"

void QESFileOutput_v2::attachDataSource(DataSource *dataSource)
{
  // std::cout << "[FILE] call attach" << std::endl;
  dataSource->attachToFile(this);
  dataSource->setOutputFields();
  m_list_data_source.push_back(dataSource);
};

void QESFileOutput_v2::notifyDataSourcesOfNewTimeEntry()
{
  // std::cout << "[FILE] notify all data sources" << std::endl;
  for (auto ds : m_list_data_source) {
    ds->notifyOfNewTimeEntry();
  }
}
void QESFileOutput_v2::save(QEStime &timeIn)
{
  // std::cout << "[FILE] call all saves" << std::endl;
  for (auto ds : m_list_data_source) {
    ds->pushToFile(timeIn);
  }
}

void QESNullOutput::attachDataSource(DataSource *dataSource)
{
  // std::cout << "[FILE] call attach" << std::endl;
  dataSource->attachToFile(this);
}
