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

/** @file SourceGeometries.h
 * @brief This class defines the interface for the source components.
 */

#pragma once

#include <random>

#include "util/QEStime.h"
#include "util/VectorMath.h"

#include "SourceComponent.h"
#include "SourceGeometries.h"


class PLUMEGeneralData;

/**
 * \brief Source Geometry: Point
 */
class PI_SourceGeometry_Point;
class SourceGeometryPoint : public SourceComponent
{
public:
  explicit SourceGeometryPoint(const vec3 &x);
  explicit SourceGeometryPoint(const PI_SourceGeometry_Point *param);
  ~SourceGeometryPoint() override = default;
  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override;

private:
  SourceGeometryPoint() = default;

  vec3 m_pos{};
};

/**
 * \brief Source Geometry: Line
 */
class PI_SourceGeometry_Line;
class SourceGeometryLine : public SourceComponent
{
public:
  SourceGeometryLine(const vec3 &pos_0, const vec3 &pos_1);
  explicit SourceGeometryLine(const PI_SourceGeometry_Line *param);

  ~SourceGeometryLine() override = default;

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override;

private:
  SourceGeometryLine() = default;
  // vec3 m_pos_start,m_pos_end,m_diff;
  vec3 m_pos_0{}, m_pos_1{}, m_diff{};

  std::random_device rd;// Will be used to obtain a seed for the random number engine
  std::mt19937 prng;// Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> uniform;
};

/**
 * \brief Source Geometry: Full Domain
 */
class PI_SourceGeometry_FullDomain;
class SourceGeometryFullDomain : public SourceComponent
{
public:
  SourceGeometryFullDomain(const PLUMEGeneralData *PGD);
  SourceGeometryFullDomain(const PI_SourceGeometry_FullDomain *param);

  ~SourceGeometryFullDomain() override = default;

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override;

private:
  SourceGeometryFullDomain() = default;

  float xDomainStart = -1.0;
  float yDomainStart = -1.0;
  float zDomainStart = -1.0;
  float xDomainEnd = -1.0;
  float yDomainEnd = -1.0;
  float zDomainEnd = -1.0;

  std::random_device rd;// Will be used to obtain a seed for the random number engine
  std::mt19937 prng;// Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> uniform;
};

/**
 * \brief Source Geometry: Spherical Shell
 */
class PI_SourceGeometry_SphereShell;
class SourceGeometrySphereShell : public SourceComponent
{
public:
  SourceGeometrySphereShell(const vec3 &min, const float &radius);
  explicit SourceGeometrySphereShell(const PI_SourceGeometry_SphereShell *param);

  ~SourceGeometrySphereShell() override = default;

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override;

private:
  SourceGeometrySphereShell() = default;

  vec3 m_x{};
  float m_radius = 0;

  std::random_device rd;// Will be used to obtain a seed for the random number engine
  std::mt19937 prng;// Standard mersenne_twister_engine seeded with rd()
  std::normal_distribution<float> normal;
};

/**
 * \brief Source Geometry: Cube
 */
class PI_SourceGeometry_Cube;
class SourceGeometryCube : public SourceComponent
{
public:
  SourceGeometryCube(const vec3 &min, const vec3 &max);
  explicit SourceGeometryCube(const PI_SourceGeometry_Cube *param);
  ~SourceGeometryCube() override = default;

  void generate(const QEStime &currTime, const int &n, QESDataTransport &data) override;

private:
  SourceGeometryCube() = default;

  vec3 m_min{};
  vec3 m_max{};

  std::random_device rd;// Will be used to obtain a seed for the random number engine
  std::mt19937 prng;// Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> uniform;
};
