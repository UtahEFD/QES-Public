"""Type stubs for the native pyQES._fire extension."""

class FireRunResult:
    fire_out: str
    plume_out: str

def run_fire(
    winds_xml_path: str,
    plume_xml_path: str = ...,
    solve_type: int = ...,
    out_basename: str = ...,
    comp_turb: bool = ...,
    fire_winds_off: bool = ...,
) -> FireRunResult: ...
