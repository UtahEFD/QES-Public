/****************************************************************************
 * Copyright (c) 2026 University of La Rochelle and TIPEE
 *
 * This file is part of QES / pyQES.
 *
 * GPL-3.0 License
 ****************************************************************************/

/** @file plume_bindings.cpp
 * @brief pybind11 bindings for the QES-Plume runner (module pyQES._plume).
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "runners/run_plume.h"

namespace py = pybind11;

PYBIND11_MODULE(_plume, m)
{
  m.doc() = "QES-Plume native bindings (pyQES).";

  py::class_<pyqes::PlumeRunResult>(m, "PlumeRunResult")
    .def_readonly("plume_out", &pyqes::PlumeRunResult::plumeOut)
    .def_readonly("particle_out", &pyqes::PlumeRunResult::particleOut)
    .def("__repr__", [](const pyqes::PlumeRunResult &r) {
      return "<PlumeRunResult plume_out='" + r.plumeOut
             + "' particle_out='" + r.particleOut + "'>";
    });

  m.def(
    "run_plume",
    [](const std::string &plume_xml_path,
       const std::string &winds_file,
       const std::string &turb_file,
       const std::string &out_basename,
       bool particle_output) {
      py::gil_scoped_release release;
      return pyqes::run_plume(plume_xml_path, winds_file, turb_file, out_basename, particle_output);
    },
    py::arg("plume_xml_path"),
    py::arg("winds_file"),
    py::arg("turb_file"),
    py::arg("out_basename") = "qes",
    py::arg("particle_output") = false,
    "Run the QES-Plume advection model on precomputed wind/turb fields.");
}
