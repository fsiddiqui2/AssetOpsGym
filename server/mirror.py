"""CouchDB as a mirror of the simulator, in the documents
AssetOpsBench's servers read. Nothing here is read back."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from ..schemas import (LaborLine, LedgerEvent, ServiceRequestEvent,
                       TelemetryRow, WorkOrderItem)
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


def work_order_item(events: list[LedgerEvent],
                    wonum: str) -> WorkOrderItem:
    """Fold one order's events into its WO document."""
    mine = [e for e in events if isinstance(e, ServiceRequestEvent)
            and e.wonum == wonum]
    if not mine:
        raise KeyError(f"no events for work order {wonum}")
    last = mine[-1]
    status_since = [b.time for a, b in zip([None, *mine], mine)
                    if a is None or a.status != b.status][-1]
    done = next((e for e in mine if e.labor_hours is not None), None)
    log = "\n".join(f"[{iso(e.time)}] {e.note}"
                    for e in mine if e.note)
    labor = last.arrive_at and [LaborLine(
        laborcode=last.tech, craft="HVAC",
        startdate=_wo_time(last.arrive_at))]
    return WorkOrderItem(
        wonum=wonum, description=last.description,
        description_longdescription=log or None, siteid=last.site,
        assetnum=last.asset, location=last.location,
        status=last.status, status_date=_wo_time(status_since),
        worktype=last.work_type, wopriority=last.priority,
        reportdate=_wo_time(mine[0].time),
        reportedby=last.reported_by, failurecode=last.code,
        actfinish=_wo_time(done.time) if done else None,
        actlabhrs=done.labor_hours if done else None,
        wplabor=labor or None,
        _id=f"wo:{last.site}:{wonum}")


def write_work_orders(couch: Couch, events: list[LedgerEvent],
                      wonums: set[str]) -> None:
    """Rewrite these orders' documents from the events."""
    for wonum in sorted(wonums):
        doc = asdict(work_order_item(events, wonum))
        old = couch.get("wo", doc["_id"])
        couch.put("wo", doc if old is None
                  else {**doc, "_rev": old["_rev"]})


def _wo_time(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S+00:00")
