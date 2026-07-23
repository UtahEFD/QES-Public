"""Convert between pydantic config models and QES XML documents.

The QES C++ parser (``ParseInterface``) reads a fixed set of XML tags. These
helpers serialize the pydantic models from :mod:`pyQES.util.config` to that exact
tag layout and parse existing QES XML files back into models.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.dom import minidom

from pydantic import BaseModel

from .config import SensorParameters, WindsParameters

__all__ = [
    "to_qes_xml",
    "from_qes_xml",
    "to_sensor_xml",
    "from_sensor_xml",
    "write_qes_xml",
    "write_sensor_xml",
]


def _format_scalar(value: Any) -> str:
    """Render a primitive as the C++ parser expects (space-padded on write)."""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _append_field(parent: ET.Element, tag: str, value: Any) -> None:
    """Append one field to ``parent`` following the QES tag conventions."""
    if value is None:
        return

    if isinstance(value, BaseModel):
        _model_to_element(value, parent, tag)
    elif isinstance(value, tuple):
        child = ET.SubElement(parent, tag)
        child.text = " ".join(_format_scalar(v) for v in value)
    elif isinstance(value, list):
        for item in value:
            _append_field(parent, tag, item)
    else:
        child = ET.SubElement(parent, tag)
        child.text = _format_scalar(value)


def _model_to_element(model: BaseModel, parent: ET.Element | None, tag: str) -> ET.Element:
    """Serialize a model into an XML element named ``tag``."""
    element = ET.Element(tag) if parent is None else ET.SubElement(parent, tag)
    for name, field in type(model).model_fields.items():
        alias = field.alias or name
        _append_field(element, alias, getattr(model, name))
    return element


def _pretty(element: ET.Element) -> str:
    """Return an indented XML string with a declaration."""
    raw = ET.tostring(element, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


def to_qes_xml(params: WindsParameters) -> str:
    """Serialize :class:`WindsParameters` to a QES-Winds XML string."""
    root = _model_to_element(params, None, "QESWindsParameters")
    return _pretty(root)


def to_sensor_xml(sensor: SensorParameters) -> str:
    """Serialize :class:`SensorParameters` to a QES sensor XML string."""
    root = _model_to_element(sensor, None, "sensor")
    return _pretty(root)


def _element_to_obj(element: ET.Element) -> Any:
    """Recursively turn an XML element into nested dict/list/str values."""
    children = [c for c in element if isinstance(c.tag, str)]
    if not children:
        return (element.text or "").strip()

    result: dict[str, Any] = {}
    for child in children:
        value = _element_to_obj(child)
        tag = child.tag
        if tag in result:
            existing = result[tag]
            if isinstance(existing, list):
                existing.append(value)
            else:
                result[tag] = [existing, value]
        else:
            result[tag] = value
    return result


def from_qes_xml(path: str | Path) -> WindsParameters:
    """Parse a QES-Winds XML file into :class:`WindsParameters`."""
    root = ET.parse(str(path)).getroot()
    return WindsParameters.model_validate(_element_to_obj(root))


def from_sensor_xml(path: str | Path) -> SensorParameters:
    """Parse a QES sensor XML file into :class:`SensorParameters`."""
    root = ET.parse(str(path)).getroot()
    return SensorParameters.model_validate(_element_to_obj(root))


def write_qes_xml(params: WindsParameters, path: str | Path) -> Path:
    """Write :class:`WindsParameters` to ``path`` and return it."""
    out = Path(path)
    out.write_text(to_qes_xml(params), encoding="utf-8")
    return out


def write_sensor_xml(sensor: SensorParameters, path: str | Path) -> Path:
    """Write :class:`SensorParameters` to ``path`` and return it."""
    out = Path(path)
    out.write_text(to_sensor_xml(sensor), encoding="utf-8")
    return out
