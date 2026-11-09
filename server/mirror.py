"""CouchDB as a mirror of the simulator, in the documents
AssetOpsBench's servers read. Nothing here is read back."""

from __future__ import annotations

from ..schemas import TelemetryRow
from .aob import Couch
from .faults import FAILURE_CODES
from .telemetry import ASSET, INSTALLED_SENSORS, SITE
from .variant import iso


def seed(couch: Couch) -> None:
    """The asset and failure-code catalogs, after recreate()."""
    couch.bulk("asset", [{
        "_id": f"asset:{ASSET}", "assetnum": ASSET,
        "description": f"{ASSET} (chilled-water plant)",
        "siteid": SITE, "orgid": "RIVPLANT", "status": "OPERATING",
        "assettype": "CHILLER", "location": "MAIN-MECH-CH6",
        "sensors": INSTALLED_SENSORS, "dataset": "asset"}])
    couch.bulk("fc", [{
        "_id": f"fc:{c.code}", "code": c.code,
        "description": c.description, "dataset": "failure_code"}
        for c in FAILURE_CODES.values()])


def write_rows(couch: Couch, rows: list[TelemetryRow]) -> None:
    couch.bulk("iot", [
        {"_id": f"iot:{ASSET}:{iso(r.timestamp)}", "asset_id": ASSET,
         "timestamp": iso(r.timestamp), **r.values} for r in rows])
