"""Type stubs for the native pyQES._plume extension."""

class PlumeRunResult:
    plume_out: str
    particle_out: str

def run_plume(
    plume_xml_path: str,
    winds_file: str,
    turb_file: str,
    out_basename: str = ...,
    particle_output: bool = ...,
) -> PlumeRunResult: ...
