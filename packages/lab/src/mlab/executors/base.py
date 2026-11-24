"""The `mlab.executor.v1` contract between the lab and a hardware sandbox.

    POST {base}/v1/jobs        {"schema": "mlab.executor.v1", "client_job_id", "task", "candidate_id",
                                "spec": <RunSpec>, "requests": [<workload rows>]}      -> {"job_id": str}
    GET  {base}/v1/jobs/{id}   -> {"job_id", "status": "queued"|"running"|"succeeded"|"failed"|"cancelled",
                                   "result": <MeasurementRecord> when succeeded, "error": str when failed}
    POST {base}/v1/jobs/{id}/cancel                                                    -> {"status": ...}

A MeasurementRecord carries one observation per request (the lab's RequestObservation fields), memory peaks,
the full hardware/software identity the simulator's calibration contract requires (docs/calibration.md in
packages/simulator), optional raw step records, and a declared source. Only `source.kind == "measured"` with a
complete identity produces `measured` evidence; anything else stays `mock`.
"""
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA = "mlab.executor.v1"
PLACEHOLDERS = {"", "unknown", "unspecified", "toy", "auto"}


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["measured", "test_fixture"]
    reference: str = Field(min_length=1)


class Identity(BaseModel):
    """What the run measured, in the terms of the simulator's profile_identity."""
    model_config = ConfigDict(extra="allow")
    model_revision: str
    devices: list[str] = Field(min_length=1)
    runtime: dict[str, str]
    weight_dtype: str
    kv_dtype: str
    activation_dtype: str
    driver: str = "not-applicable"

    @model_validator(mode="after")
    def explicit(self):
        values = [self.model_revision, self.weight_dtype, self.kv_dtype, self.activation_dtype, *self.devices,
                  self.runtime.get("name", ""), self.runtime.get("version", "")]
        if any(str(value).strip().lower() in PLACEHOLDERS for value in values):
            raise ValueError("measured identity needs explicit model revision, devices, runtime and dtypes")
        return self


class MeasurementRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    observations: list[dict[str, Any]] = Field(min_length=1)
    vram_peak_gib: float | None = Field(default=None, ge=0)
    kv_peak_gib: float | None = Field(default=None, ge=0)
    identity: Identity
    source: Source
    step_records: list[dict[str, Any]] = []
    logs: list[str] = []

    @property
    def evidence_level(self):
        return "measured" if self.source.kind == "measured" else "mock"


class HardwareExecutor(Protocol):
    name: str

    def execute(self, root, run, spec, requests) -> MeasurementRecord:
        """Run one measurement to completion (or raise LabError); never retry a submitted job."""
        ...
