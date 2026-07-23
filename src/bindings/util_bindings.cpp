/****************************************************************************
 * Copyright (c) 2026 University of La Rochelle and TIPEE
 *
 * This file is part of QES / pyQES.
 *
 * GPL-3.0 License
 ****************************************************************************/

/** @file util_bindings.cpp
 * @brief pybind11 bindings for shared QES metadata (module pyQES._util).
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

#ifndef QES_VERSION_INFO
#define QES_VERSION_INFO "unknown"
#endif

PYBIND11_MODULE(_util, m)
{
  m.doc() = "QES shared native bindings (pyQES).";

  m.attr("qes_version") = QES_VERSION_INFO;

#ifdef HAS_CUDA
  m.attr("has_cuda") = true;
#else
  m.attr("has_cuda") = false;
#endif
}
