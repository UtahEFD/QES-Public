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
 * @file FuelProperties.hpp
 * @brief This class stores fuel properties of Anderson 13 / Scott and Burgan 40 fuels
**/
class FuelProperties
{

public:
  float windrf;
  float fgi;
  float fueldepthm;
  int savr;
  float fuelmce;
  int fueldens;
  float st;
  float se;
  int weight;
  float fci_d;
  int fct;
  int ichap;
  float fci;
  float fcbr;
  int hfgl;
  int cmbcnst;
  float fuelheat;
  float fuelmc_g;
  int fuelmc_c;
  float slope;

  FuelProperties(float windrf, float fgi, float fueldepthm, int savr, float fuelmce, int fueldens, float st, float se, int weight, float fci_d, int fct, int ichap, float fci, float fcbr, int hfgl, int cmbcnst, float fuelheat, float fuelmc_g, int fuelmc_c, float slope) : windrf(windrf), fgi(fgi), fueldepthm(fueldepthm), savr(savr), fuelmce(fuelmce),
                                                                                                                                                                                                                                                                               fueldens(fueldens), st(st), se(se), weight(weight), fci_d(fci_d),
                                                                                                                                                                                                                                                                               fct(fct), ichap(ichap), fci(fci), fcbr(fcbr), hfgl(hfgl), cmbcnst(cmbcnst),
                                                                                                                                                                                                                                                                               fuelheat(fuelheat), fuelmc_g(fuelmc_g), fuelmc_c(fuelmc_c), slope(slope) {}

private:
};

// Short grass (1 ft)
class ShortGrass : public FuelProperties
{
public:
  ShortGrass() : FuelProperties(0.3600, 0.1660, 0.3050, 3500, 0.1200, 32, 0.0555, 0.0100, 7, 0.0000, 60, 0, 0.0000, 0.0000, 170000, 17433000, 7496.2, 0.0650, 1, 0.0) {}
};
// Timber (grass and understory)
class TimberGrass : public FuelProperties
{
public:
  TimberGrass() : FuelProperties(0.3600, 0.8970, 0.3050, 2784, 0.1500, 32, 0.0555, 0.0100, 7, 0.0000, 60, 0, 0.0000, 0.0000, 170000, 17433000, 7496.2, 0.0650, 1, 0.0) {}
};
// Tall grass (2.5 ft)
class TallGrass : public FuelProperties
{
public:
  TallGrass() : FuelProperties(0.4400, 0.6750, 0.7620, 1500, 0.2500, 32, 0.0555, 0.0100, 7, 0.0000, 60, 0, 0.0000, 0.0000, 170000, 17433000, 7496.2, 0.0650, 1, 0.0) {}
};
// Chaparral (6 ft)
class Chaparral : public FuelProperties
{
public:
  Chaparral() : FuelProperties(0.5500, 2.4680, 1.8290, 1739, 0.2000, 32, 0.0555, 0.0100, 180, 1.1230, 60, 1, 2.2460, 0.0187, 170000, 17433000, 7496.2, 0.0650, 1, 0.0) {}
};
// Brush (2 ft)
class Brush : public FuelProperties
{
public:
  Brush() : FuelProperties(0.4200, 0.7850, 0.6100, 1683, 0.2000, 32, 0.0555, 0.0100, 100, 0.0000, 60, 0, 0.0000, 0.0000, 170000, 17433000, 7496.2, 0.0650, 1, 0.0) {}
};
// Dormant brush, hardwood slash
class DormantBrush : public FuelProperties
{
public:
  DormantBrush() : FuelProperties(0.4400, 1.3450, 0.7620, 1564, 0.2500, 32, 0.0555, 0.0100, 100, 0.0000, 60, 0, 0.0000, 0.0000, 170000, 17433000, 7496.2, 0.0650, 1, 0.0) {}
};
// Southern rough
class SouthernRough : public FuelProperties
{
public:
  SouthernRough() : FuelProperties(0.4400, 1.0920, 0.7620, 1562, 0.4000, 32, 0.0555, 0.0100, 100, 0.0000, 60, 0, 0.0000, 0.0000, 170000, 17433000, 7496.2, 0.0650, 1, 0.0) {}
};
// Closed timber litter
class TimberClosedLitter : public FuelProperties
{
public:
  TimberClosedLitter() : FuelProperties(0.3600, 1.1210, 0.0610, 1889, 0.3000, 32, 0.0555, 0.0100, 900, 1.1210, 60, 0, 2.2420, 0.0187, 170000, 17433000, 7496.2, 0.0650, 1, 0.0) {}
};
// Hardwood litter
class HarwoodLitter : public FuelProperties
{
public:
  HarwoodLitter() : FuelProperties(0.3600, 0.7800, 0.0610, 2484, 0.2500, 32, 0.0555, 0.0100, 900, 1.1210, 120, 0, 2.2420, 0.0093, 170000, 17433000, 7496.2, 0.0650, 1, 0.0) {}
};
// Timber (litter + understory)
class TimberLitter : public FuelProperties
{
public:
  TimberLitter() : FuelProperties(0.3600, 2.6940, 0.3050, 1764, 0.2500, 32, 0.0555, 0.0100, 900, 1.1210, 180, 0, 2.2420, 0.0062, 170000, 17433000, 7496.2, 0.0650, 1, 0.0) {}
};
// Light logging slash
class LoggingSlashLight : public FuelProperties
{
public:
  LoggingSlashLight() : FuelProperties(0.3600, 2.5820, 0.3050, 1182, 0.1500, 32, 0.0555, 0.0100, 900, 1.1210, 180, 0, 2.2420, 0.0062, 170000, 17433000, 7496.2, 0.0650, 1, 0.0) {}
};
// Medium logging slash
class LoggingSlashMedium : public FuelProperties
{
public:
  LoggingSlashMedium() : FuelProperties(0.4300, 7.7490, 0.7010, 1145, 0.2000, 32, 0.0555, 0.0100, 900, 1.1210, 180, 0, 2.2420, 0.0062, 170000, 17433000, 7496.2, 0.0650, 1, 0.0) {}
};
// Heavy logging slash
class LoggingSlashHeavy : public FuelProperties
{
public:
  LoggingSlashHeavy() : FuelProperties(0.4600, 13.0240, 0.9140, 1159, 0.2500, 32, 0.0555, 0.0100, 900, 1.1210, 180, 0, 2.2420, 0.0062, 170000, 17433000, 7496.2, 0.0650, 1, 0.0) {}
};
// Urban/Developed
class Urban : public FuelProperties
{
public:
  Urban() : FuelProperties(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {}
};
// Snow/Ice
class Snow : public FuelProperties
{
public:
  Snow() : FuelProperties(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {}
};
// Agricultural
class Agricultural : public FuelProperties
{
public:
  Agricultural() : FuelProperties(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {}
};
// Open Water
class Water : public FuelProperties
{
public:
  Water() : FuelProperties(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {}
};
// Bare Ground
class Bare : public FuelProperties
{
public:
  Bare() : FuelProperties(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {}
};
