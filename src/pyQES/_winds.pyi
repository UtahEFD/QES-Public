"""Type stubs for the native pyQES._winds extension."""

class WindsRunResult:
    winds_out: str
    winds_wk: str
    turb_out: str

def run_winds(
    xml_path: str,
    solve_type: int = ...,
    out_basename: str = ...,
    visu_output: bool = ...,
    wksp_output: bool = ...,
    turb_output: bool = ...,
) -> WindsRunResult: ...
