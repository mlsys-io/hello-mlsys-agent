# Simulator calibration ledger

Real GPU sandbox runs come back here and improve the simulator, behind a gate that never lets a worse Profile
replace a better one.

```
mlab measure (--hardware remote) ──► runs/**/measurement.json   (step_records + identity.simulator_identity)
        │ scripts/calibration/loop.py export WS... --out DIR
        ▼
simulator step dataset  ──ingest──►  datasets/<identity>/<sha>.json  + frozen .train/.holdout split + ledger.jsonl
        │ promote <sha>: fit a candidate on train; score it and the CURRENT Profile on the same holdout
        ▼
profiles/<identity>/vN.json + CURRENT     (only when budgets.yaml passes and nothing regresses)
```

- **Inputs.** A sandbox's `MeasurementRecord` (`packages/lab/src/mlab/executors/base.py`) must carry
  `step_records` in the simulator's observation schema (`record_id`, `group_id`, `phase`, `stage_index`,
  `query_tokens`, `context_lengths`, `duration_ms`; see `packages/simulator/docs/calibration.md`) and
  `identity.simulator_identity`, the full simulator identity: model geometry and revision, devices, parallelism,
  topology, runtime versions and timing boundary, KV block size. Records with different identities become
  different datasets and Profiles.
- **Holdout, frozen at ingest.** Groups named `holdout*` are the campaign's declared holdout; otherwise the
  simulator's seeded grouped split (`budgets.yaml`) is used. Either way the split is written once, hashed into
  the ledger and never refitted. A held-out shape must lie *between* measured grid values (for example, a context
  length the training grid did not measure). A shape exactly on grid values needs itself as a corner, and the
  simulator refuses to extrapolate.
- **Gate.** The candidate must cover `min_coverage` of the holdout within `max_mape_percent` (step-time MAPE from
  `mlsys-sim validate`), and must not be worse than the current Profile on the same holdout by more than
  `regression_tolerance_percent`. A refusal is recorded in the ledger with its reasons.
- **Review.** Run `ingest` and `promote` locally and commit the new files in a PR. CI runs `loop.py verify`
  (every dataset, split and Profile matches its ledger hash, and CURRENT names the newest promotion) and the loop
  tests.
- **Fixtures.** Test-fixture data needs `--allow-test-fixture` at every step, and the simulator labels its
  Profiles as fixtures. Only `measured` data belongs in this directory on `main`.

```bash
python scripts/calibration/loop.py export ~/labs/ws1 ~/labs/ws2 --out /tmp/datasets
python scripts/calibration/loop.py ingest /tmp/datasets/<key>.json
python scripts/calibration/loop.py promote <sha-prefix>     # exit 1 when refused
python scripts/calibration/loop.py verify
```

Not wired yet: the lab's `mlab.simulator.v1` agent API runs the simulator's uncalibrated analytical model. Using
a promoted Profile there (`calibration.backend="profile"`) is the next integration step, once a sandbox produces
measured data for the exercise's model and hardware identities.
