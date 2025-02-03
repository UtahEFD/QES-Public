/****************************************************************************
 * Copyright (c) 2025 University of Utah
 * Copyright (c) 2025 University of Minnesota Duluth
 *
 * Copyright (c) 2025 Matthew Moody
 * Copyright (c) 2025 Jeremy Gibbs
 * Copyright (c) 2025 Rob Stoll
 * Copyright (c) 2025 Fabien Margairaz
 * Copyright (c) 2025 Brian Bailey
 * Copyright (c) 2025 Pete Willemsen
 *
 * This file is part of QES-Fire
 *
 * GPL-3.0 License
 *
 * QES-Fire is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Fire is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
/**
 * @file FuelMap.cpp
 * @brief Sets fuel in firemap from GEOTIF if fuel file is specified, or as a constant fuel type if no GEOTIF
 */
#include "Fire.h"

using namespace std;

void Fire ::FuelMap(WINDSInputData *WID, WINDSGeneralData *WGD)
{
  // Set fuel properties for domain
  fuel_type = WID->fires->fuelType;
  std::string fuelFile = WID->fires->fuelFile;
  FuelRead *Fuel_read = nullptr;

  Vector3Int domain;
  Vector3 grid;
  domain = WID->simParams->domain;
  grid = WID->simParams->grid;
  float halo_x = WID->simParams->halo_x;
  float halo_y = WID->simParams->halo_y;
  if (fuelFile != "") {
    Fuel_read = new FuelRead(fuelFile,
                             std::tuple<int, int>(domain[0], domain[1]),
                             std::tuple<float, float>(grid[0], grid[1]));

    int fID = fuel_type;
    int fuelID = fuel_type;
    for (int j = 0; j < ny - 1; j++) {
      for (int i = 0; i < nx - 1; i++) {
        int idx = i + j * (nx - 1);

        fID = Fuel_read->fuelField[idx];
        fuel_map[idx] = fID;
        fuelID = fID;

        if (fuelID == 1)
          fire_cells[idx].fuel = new ShortGrass();
        else if (fuelID == 2)
          fire_cells[idx].fuel = new TimberGrass();
        else if (fuelID == 3)
          fire_cells[idx].fuel = new TallGrass();
        else if (fuelID == 4)
          fire_cells[idx].fuel = new Chaparral();
        else if (fuelID == 5)
          fire_cells[idx].fuel = new Brush();
        else if (fuelID == 6)
          fire_cells[idx].fuel = new DormantBrush();
        else if (fuelID == 7)
          fire_cells[idx].fuel = new SouthernRough();
        else if (fuelID == 8)
          fire_cells[idx].fuel = new TimberClosedLitter();
        else if (fuelID == 9)
          fire_cells[idx].fuel = new HarwoodLitter();
        else if (fuelID == 10)
          fire_cells[idx].fuel = new TimberLitter();
        else if (fuelID == 11)
          fire_cells[idx].fuel = new LoggingSlashLight();
        else if (fuelID == 12)
          fire_cells[idx].fuel = new LoggingSlashMedium();
        else if (fuelID == 13)
          fire_cells[idx].fuel = new LoggingSlashHeavy();
        else if (fuelID == 91)
          fire_cells[idx].fuel = new Urban();
        else if (fuelID == 92)
          fire_cells[idx].fuel = new Snow();
        else if (fuelID == 93)
          fire_cells[idx].fuel = new Agricultural();
        else if (fuelID == 98)
          fire_cells[idx].fuel = new Water();
        else if (fuelID == 99)
          fire_cells[idx].fuel = new Bare();
        else if (fuelID == 101)
          fire_cells[idx].fuel = new GR1();
        else if (fuelID == 102)
          fire_cells[idx].fuel = new GR2();
        else if (fuelID == 103)
          fire_cells[idx].fuel = new GR3();
        else if (fuelID == 104)
          fire_cells[idx].fuel = new GR4();
        else if (fuelID == 105)
          fire_cells[idx].fuel = new GR5();
        else if (fuelID == 106)
          fire_cells[idx].fuel = new GR6();
        else if (fuelID == 107)
          fire_cells[idx].fuel = new GR7();
        else if (fuelID == 108)
          fire_cells[idx].fuel = new GR8();
        else if (fuelID == 109)
          fire_cells[idx].fuel = new GR9();
        else if (fuelID == 121)
          fire_cells[idx].fuel = new GS1();
        else if (fuelID == 122)
          fire_cells[idx].fuel = new GS2();
        else if (fuelID == 123)
          fire_cells[idx].fuel = new GS3();
        else if (fuelID == 124)
          fire_cells[idx].fuel = new GS4();
        else if (fuelID == 141)
          fire_cells[idx].fuel = new SH1();
        else if (fuelID == 142)
          fire_cells[idx].fuel = new SH2();
        else if (fuelID == 143)
          fire_cells[idx].fuel = new SH3();
        else if (fuelID == 144)
          fire_cells[idx].fuel = new SH4();
        else if (fuelID == 145)
          fire_cells[idx].fuel = new SH5();
        else if (fuelID == 146)
          fire_cells[idx].fuel = new SH6();
        else if (fuelID == 147)
          fire_cells[idx].fuel = new SH7();
        else if (fuelID == 148)
          fire_cells[idx].fuel = new SH8();
        else if (fuelID == 149)
          fire_cells[idx].fuel = new SH9();
        else if (fuelID == 161)
          fire_cells[idx].fuel = new TU1();
        else if (fuelID == 162)
          fire_cells[idx].fuel = new TU2();
        else if (fuelID == 163)
          fire_cells[idx].fuel = new TU3();
        else if (fuelID == 164)
          fire_cells[idx].fuel = new TU4();
        else if (fuelID == 165)
          fire_cells[idx].fuel = new TU5();
        else if (fuelID == 181)
          fire_cells[idx].fuel = new TL1();
        else if (fuelID == 182)
          fire_cells[idx].fuel = new TL2();
        else if (fuelID == 183)
          fire_cells[idx].fuel = new TL3();
        else if (fuelID == 184)
          fire_cells[idx].fuel = new TL4();
        else if (fuelID == 185)
          fire_cells[idx].fuel = new TL5();
        else if (fuelID == 186)
          fire_cells[idx].fuel = new TL6();
        else if (fuelID == 187)
          fire_cells[idx].fuel = new TL7();
        else if (fuelID == 188)
          fire_cells[idx].fuel = new TL8();
        else if (fuelID == 189)
          fire_cells[idx].fuel = new TL9();
        else if (fuelID == 201)
          fire_cells[idx].fuel = new SB1();
        else if (fuelID == 202)
          fire_cells[idx].fuel = new SB2();
        else if (fuelID == 203)
          fire_cells[idx].fuel = new SB3();
        else if (fuelID == 204)
          fire_cells[idx].fuel = new SB4();
        else
          fire_cells[idx].fuel = new SB4();
      }
    }
    std::cout << "Fuel set" << std::endl;
  } else {
    for (int j = 0; j < ny - 1; j++) {
      for (int i = 0; i < nx - 1; i++) {
        int idx = i + j * (nx - 1);
        fuel_map[idx] = fuel_type;
        if (fuel_type == 1) fire_cells[idx].fuel = new ShortGrass();
        if (fuel_type == 2) fire_cells[idx].fuel = new TimberGrass();
        if (fuel_type == 3) fire_cells[idx].fuel = new TallGrass();
        if (fuel_type == 4) fire_cells[idx].fuel = new Chaparral();
        if (fuel_type == 5) fire_cells[idx].fuel = new Brush();
        if (fuel_type == 6) fire_cells[idx].fuel = new DormantBrush();
        if (fuel_type == 7) fire_cells[idx].fuel = new SouthernRough();
        if (fuel_type == 8) fire_cells[idx].fuel = new TimberClosedLitter();
        if (fuel_type == 9) fire_cells[idx].fuel = new HarwoodLitter();
        if (fuel_type == 10) fire_cells[idx].fuel = new TimberLitter();
        if (fuel_type == 11) fire_cells[idx].fuel = new LoggingSlashLight();
        if (fuel_type == 12) fire_cells[idx].fuel = new LoggingSlashMedium();
        if (fuel_type == 13) fire_cells[idx].fuel = new LoggingSlashHeavy();
        if (fuel_type == 91) fire_cells[idx].fuel = new Urban();
        if (fuel_type == 92) fire_cells[idx].fuel = new Snow();
        if (fuel_type == 93) fire_cells[idx].fuel = new Agricultural();
        if (fuel_type == 98) fire_cells[idx].fuel = new Water();
        if (fuel_type == 99) fire_cells[idx].fuel = new Bare();
        if (fuel_type == 101) fire_cells[idx].fuel = new GR1();
        if (fuel_type == 102) fire_cells[idx].fuel = new GR2();
        if (fuel_type == 103) fire_cells[idx].fuel = new GR3();
        if (fuel_type == 104) fire_cells[idx].fuel = new GR4();
        if (fuel_type == 105) fire_cells[idx].fuel = new GR5();
        if (fuel_type == 106) fire_cells[idx].fuel = new GR6();
        if (fuel_type == 107) fire_cells[idx].fuel = new GR7();
        if (fuel_type == 108) fire_cells[idx].fuel = new GR8();
        if (fuel_type == 109) fire_cells[idx].fuel = new GR9();
        if (fuel_type == 121) fire_cells[idx].fuel = new GS1();
        if (fuel_type == 122) fire_cells[idx].fuel = new GS2();
        if (fuel_type == 123) fire_cells[idx].fuel = new GS3();
        if (fuel_type == 124) fire_cells[idx].fuel = new GS4();
        if (fuel_type == 141) fire_cells[idx].fuel = new SH1();
        if (fuel_type == 142) fire_cells[idx].fuel = new SH2();
        if (fuel_type == 143) fire_cells[idx].fuel = new SH3();
        if (fuel_type == 144) fire_cells[idx].fuel = new SH4();
        if (fuel_type == 145) fire_cells[idx].fuel = new SH5();
        if (fuel_type == 146) fire_cells[idx].fuel = new SH6();
        if (fuel_type == 147) fire_cells[idx].fuel = new SH7();
        if (fuel_type == 148) fire_cells[idx].fuel = new SH8();
        if (fuel_type == 149) fire_cells[idx].fuel = new SH9();
        if (fuel_type == 161) fire_cells[idx].fuel = new TU1();
        if (fuel_type == 162) fire_cells[idx].fuel = new TU2();
        if (fuel_type == 163) fire_cells[idx].fuel = new TU3();
        if (fuel_type == 164) fire_cells[idx].fuel = new TU4();
        if (fuel_type == 165) fire_cells[idx].fuel = new TU5();
        if (fuel_type == 181) fire_cells[idx].fuel = new TL1();
        if (fuel_type == 182) fire_cells[idx].fuel = new TL2();
        if (fuel_type == 183) fire_cells[idx].fuel = new TL3();
        if (fuel_type == 184) fire_cells[idx].fuel = new TL4();
        if (fuel_type == 185) fire_cells[idx].fuel = new TL5();
        if (fuel_type == 186) fire_cells[idx].fuel = new TL6();
        if (fuel_type == 187) fire_cells[idx].fuel = new TL7();
        if (fuel_type == 188) fire_cells[idx].fuel = new TL8();
        if (fuel_type == 189) fire_cells[idx].fuel = new TL9();
        if (fuel_type == 201) fire_cells[idx].fuel = new SB1();
        if (fuel_type == 202) fire_cells[idx].fuel = new SB2();
        if (fuel_type == 203) fire_cells[idx].fuel = new SB3();
        if (fuel_type == 204) fire_cells[idx].fuel = new SB4();
      }
    }
    std::cout << "fuel set constant = " << fuel_type << std::endl;
  }
}