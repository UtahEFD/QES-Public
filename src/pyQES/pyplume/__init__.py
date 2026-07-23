"""QES-Plume Python interface.

Runs the Lagrangian plume advection model on precomputed QES-Winds workspace and
QES-Turb NetCDF fields. The plume parameters are supplied through a QES-Plume XML
file (the plume XML schema is large; the model is passed through to the solver).
"""

from __future__ import annotations

from pathlib import Path

from .._extensions import load_extension
from ..util.paths import resolve_work_dir

__all__ = ["run"]


def run(
    *,
    xml: str | Path,
    winds_file: str | Path,
    turb_file: str | Path,
    out_basename: str = "qes",
    particle_output: bool = False,
    work_dir: str | Path | None = None,
):
    """Run QES-Plume and return the output NetCDF paths.

    :param xml: Path to the QES-Plume XML parameter file.
    :param winds_file: QES-Winds workspace NetCDF (``*_windsWk.nc``).
    :param turb_file: QES-Turb NetCDF (``*_turbOut.nc``).
    :param out_basename: Basename for the produced NetCDF files.
    :param particle_output: Also write the debug Lagrangian particle file.
    :param work_dir: Directory for outputs (a temp dir is used if omitted).
    :returns: The native ``PlumeRunResult`` (``plume_out``/``particle_out``).
    """
    ext = load_extension("_plume")
    work = resolve_work_dir(work_dir)
    out_prefix = str(work / out_basename)

    return ext.run_plume(
        plume_xml_path=str(Path(xml).resolve()),
        winds_file=str(Path(winds_file).resolve()),
        turb_file=str(Path(turb_file).resolve()),
        out_basename=out_prefix,
        particle_output=particle_output,
    )
