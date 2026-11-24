"""The simulator improvement loop: sandbox measurements -> datasets -> gated, versioned simulator Profiles.

    loop.py export WORKSPACE... --out DIR     lab runs with measurement.json -> simulator step datasets (one per identity)
    loop.py ingest DATASET.json               add to calibration/datasets/, freeze its holdout split, append the ledger
                                              (groups named holdout* are the declared holdout; else a seeded grouped split)
    loop.py evaluate SHA                      fit a candidate on the train split; score it and the current Profile on the holdout
    loop.py promote SHA                       evaluate, and on a pass write calibration/profiles/<key>/vN.json + CURRENT
    loop.py verify                            CI: every ledger entry's files and Profiles match their recorded hashes
                                              and CURRENT names the newest promotion (it does not refit)

Holdout records never enter a fit. The current Profile is replaced only by a candidate that meets the budgets in
calibration/budgets.yaml and does not regress on the same holdout. Test-fixture data needs --allow-test-fixture
at every step, and its Profiles are labelled as fixtures by the simulator itself.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def simulator(allow_fixture):
    """Run one `mlsys-sim` subcommand in this interpreter; the parsed status line, or SystemExit on failure."""
    def sim(*args):
        flags = ["--allow-test-fixture"] if allow_fixture else []
        done = subprocess.run([sys.executable, "-m", "mlsys_sim", *args, *flags], capture_output=True, text=True)
        try:
            status = json.loads(done.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            status = {}
        if done.returncode != 0 or status.get("status") != "COMPLETED":
            raise SystemExit(f"mlsys-sim {args[0]} failed: {(done.stdout + done.stderr).strip()[-600:]}")
        return status
    return sim


def gate(train, holdout, current, version, budgets, allow_fixture=False, keep=None):
    """Stateless promotion gate: fit a candidate Profile on `train`, score it and the `current` Profile (or None)
    on the same `holdout`, and apply `budgets`. `keep` receives the fitted Profile. Used by the ledger below and
    by the `calibrate` job in mlab.jobs."""
    sim = simulator(allow_fixture)
    holdout_records = len(json.loads(Path(holdout).read_text())["records"])
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)

        def score(profile, name):
            sim("validate", "--dataset", str(holdout), "--profile", str(profile), "--out", str(scratch / name))
            return json.loads((scratch / name / "result.json").read_text())["summary"]

        parent = ["--parent-profile", str(current)] if current else []
        sim("calibrate", "--dataset", str(train), "--profile-version", version, *parent, "--out", str(scratch / "fit"))
        candidate = score(scratch / "fit/profile.json", "candidate")
        baseline = score(current, "current") if current else None
        if keep:
            shutil.copyfile(scratch / "fit/profile.json", keep)
    reasons = []
    if holdout_records < budgets["min_holdout_records"]:
        reasons.append(f"holdout has {holdout_records} records; need {budgets['min_holdout_records']}")
    coverage, mape = candidate["coverage"] or 0.0, candidate["mape_percent"]
    if coverage < budgets["min_coverage"]:
        reasons.append(f"candidate covers {coverage:.0%} of the holdout; need {budgets['min_coverage']:.0%}")
    if mape is None or mape > budgets["max_mape_percent"]:
        reasons.append(f"candidate holdout MAPE {mape}% exceeds {budgets['max_mape_percent']}%")
    if baseline and baseline["mape_percent"] is not None and mape is not None and \
            mape > baseline["mape_percent"] + budgets["regression_tolerance_percent"]:
        reasons.append(f"candidate MAPE {mape:.2f}% regresses on the current Profile's {baseline['mape_percent']:.2f}%")
    if baseline and coverage + 1e-9 < (baseline["coverage"] or 0.0):
        reasons.append(f"candidate coverage {coverage:.0%} is below the current Profile's {baseline['coverage']:.0%}")
    return {"version": version, "candidate": candidate, "current_on_holdout": baseline,
            "promote": not reasons, "reasons": reasons}


class Loop:
    def __init__(self, root=None, allow_fixture=False):
        self.root = Path(root or REPO / "calibration")
        self.allow_fixture = allow_fixture
        self.budgets = yaml.safe_load((self.root / "budgets.yaml").read_text())

    # -- helpers ---------------------------------------------------------------------------------------------
    def sim(self, *args):
        return simulator(self.allow_fixture)(*args)

    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def identity_key(identity):
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()[:16]

    def ledger(self):
        path = self.root / "ledger.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

    def append(self, entry):
        with (self.root / "ledger.jsonl").open("a") as stream:
            stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}, sort_keys=True) + "\n")

    def entry(self, sha):
        found = [e for e in self.ledger() if e["event"] == "ingest" and e["sha256"].startswith(sha)]
        if len(found) != 1:
            raise SystemExit(f"no single ingested dataset matches {sha}")
        return found[0]

    def current(self, key):
        marker = self.root / "profiles" / key / "CURRENT"
        return (self.root / "profiles" / key / (marker.read_text().strip() + ".json")) if marker.exists() else None

    # -- export ----------------------------------------------------------------------------------------------
    def export(self, workspaces, out):
        """Sandbox step records from lab runs, grouped by the full simulator identity the sandbox declared."""
        groups = {}
        for workspace in map(Path, workspaces):
            for record_path in sorted(workspace.rglob("measurement.json")):
                record = json.loads(record_path.read_text())
                kind = record["source"]["kind"]
                if kind != "measured" and not self.allow_fixture:
                    continue
                identity = record["identity"].get("simulator_identity")
                if not identity or not record.get("step_records"):
                    continue
                # Run ids restart in every workspace, so the prefix carries the workspace too.
                run = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:8] + "/" + record_path.parent.name
                key = self.identity_key(identity)
                bucket = groups.setdefault(key, {"identity": identity, "kinds": set(), "references": [], "records": []})
                bucket["kinds"].add(kind)
                bucket["references"].append(record["source"]["reference"])
                for index, step in enumerate(record["step_records"]):
                    bucket["records"].append({**step, "record_id": f"{run}:{step.get('record_id', index)}",
                                              "group_id": f"{run}:{step.get('group_id', index)}"})
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for key, bucket in groups.items():
            kind = "measured" if bucket["kinds"] == {"measured"} else "test_fixture"
            dataset = {"schema_version": 1, "dataset_id": f"sandbox-{key}-{len(bucket['records'])}", "identity": bucket["identity"],
                       "source": {"kind": kind, "reference": "; ".join(sorted(set(bucket["references"])))[:2000]},
                       "records": bucket["records"]}
            path = out / f"{key}.json"
            path.write_text(json.dumps(dataset, indent=1, sort_keys=True) + "\n")
            written.append(str(path))
        return {"datasets": written}

    # -- ingest ----------------------------------------------------------------------------------------------
    def ingest(self, dataset):
        data = json.loads(Path(dataset).read_text())
        key = self.identity_key(data["identity"])
        sha = self.digest(dataset)
        if any(e["event"] == "ingest" and e["sha256"] == sha for e in self.ledger()):
            raise SystemExit(f"dataset {sha[:12]} is already in the ledger")
        folder = self.root / "datasets" / key
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{sha[:16]}.json"
        shutil.copyfile(dataset, target)
        (self.root / "profiles" / key).mkdir(parents=True, exist_ok=True)
        (self.root / "profiles" / key / "identity.json").write_text(json.dumps(data["identity"], indent=1, sort_keys=True) + "\n")
        # A measurement campaign may declare its holdout (groups named holdout*, e.g. interior shapes whose
        # interpolation corners were all measured); otherwise the simulator's random grouped split is used.
        declared = sorted({r["group_id"] for r in data["records"] if r["group_id"].split(":")[-1].startswith("holdout")})
        split = (["--holdout-group", *[g for group in declared for g in ("--holdout-group", group)][1:]] if declared else
                 ["--holdout-fraction", str(self.budgets["holdout_fraction"]), "--seed", str(self.budgets["split_seed"])])
        with tempfile.TemporaryDirectory() as scratch:  # the simulator's own split, done once and frozen
            self.sim("validate", "--dataset", str(target), "--profile-version", "split-only", *split,
                     "--out", str(Path(scratch) / "split"))
            for part in ("train", "holdout"):
                shutil.copyfile(Path(scratch) / "split" / f"{part}.json", folder / f"{sha[:16]}.{part}.json")
        holdout = json.loads((folder / f"{sha[:16]}.holdout.json").read_text())["records"]
        entry = {"event": "ingest", "sha256": sha, "identity_key": key, "source": data["source"],
                 "records": len(data["records"]), "holdout_records": len(holdout),
                 "split": {**({"declared_groups": declared} if declared else
                              {"fraction": self.budgets["holdout_fraction"], "seed": self.budgets["split_seed"]}),
                           "train_sha256": self.digest(folder / f"{sha[:16]}.train.json"),
                           "holdout_sha256": self.digest(folder / f"{sha[:16]}.holdout.json")}}
        self.append(entry)
        return entry

    # -- evaluate / promote ----------------------------------------------------------------------------------
    def files(self, entry):
        folder = self.root / "datasets" / entry["identity_key"]
        stem = entry["sha256"][:16]
        return folder / f"{stem}.train.json", folder / f"{stem}.holdout.json"

    def evaluate(self, sha, keep=None):
        entry = self.entry(sha)
        train, holdout = self.files(entry)
        for path, expected in ((train, entry["split"]["train_sha256"]), (holdout, entry["split"]["holdout_sha256"])):
            if self.digest(path) != expected:
                raise SystemExit(f"{path.name} changed since ingest; the holdout split is frozen")
        current = self.current(entry["identity_key"])
        versions = [e for e in self.ledger() if e["event"] == "promote" and e["identity_key"] == entry["identity_key"]]
        decision = gate(train, holdout, current, f"v{len(versions) + 1}", self.budgets, self.allow_fixture, keep)
        return {"dataset": entry["sha256"][:16], "identity_key": entry["identity_key"],
                "current": current.stem if current else None, **decision}

    def promote(self, sha):
        entry = self.entry(sha)
        folder = self.root / "profiles" / entry["identity_key"]
        with tempfile.TemporaryDirectory() as scratch:
            fitted = Path(scratch) / "profile.json"
            decision = self.evaluate(sha, keep=fitted)
            if decision["promote"]:
                shutil.copyfile(fitted, folder / f"{decision['version']}.json")
                (folder / "CURRENT").write_text(decision["version"] + "\n")
        self.append({"event": "promote" if decision["promote"] else "refuse", "identity_key": entry["identity_key"],
                     "dataset_sha256": entry["sha256"], "version": decision["version"], "replaces": decision["current"],
                     "candidate": decision["candidate"], "current_on_holdout": decision["current_on_holdout"],
                     "reasons": decision["reasons"],
                     **({"profile_sha256": self.digest(folder / f"{decision['version']}.json")} if decision["promote"] else {})})
        return decision

    def verify(self):
        """Every ingested file is intact; every promoted Profile file matches its ledger hash and CURRENT points at
        the newest promotion for its identity."""
        problems = []
        ledger = self.ledger()
        for entry in (e for e in ledger if e["event"] == "ingest"):
            folder = self.root / "datasets" / entry["identity_key"]
            stem = entry["sha256"][:16]
            if self.digest(folder / f"{stem}.json") != entry["sha256"]:
                problems.append(f"dataset {stem} changed")
            for part in ("train", "holdout"):
                if self.digest(folder / f"{stem}.{part}.json") != entry["split"][f"{part}_sha256"]:
                    problems.append(f"{part} split of {stem} changed")
        latest = {}
        for entry in (e for e in ledger if e["event"] == "promote"):
            path = self.root / "profiles" / entry["identity_key"] / f"{entry['version']}.json"
            if not path.exists() or self.digest(path) != entry["profile_sha256"]:
                problems.append(f"profile {entry['identity_key']}/{entry['version']} missing or changed")
            latest[entry["identity_key"]] = entry["version"]
        for key, version in latest.items():
            marker = self.root / "profiles" / key / "CURRENT"
            if not marker.exists() or marker.read_text().strip() != version:
                problems.append(f"profiles/{key}/CURRENT should name {version}")
        return {"ok": not problems, "problems": problems, "ingested": sum(e["event"] == "ingest" for e in ledger),
                "promoted": len([e for e in ledger if e["event"] == "promote"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, help="calibration directory (default: the repository's calibration/)")
    parser.add_argument("--allow-test-fixture", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("workspaces", nargs="+")
    export.add_argument("--out", required=True)
    for name in ("ingest",):
        commands.add_parser(name).add_argument("dataset")
    for name in ("evaluate", "promote"):
        commands.add_parser(name).add_argument("sha")
    commands.add_parser("verify")
    args = parser.parse_args(argv)
    loop = Loop(args.root, args.allow_test_fixture)
    if args.command == "export":
        result = loop.export(args.workspaces, args.out)
    elif args.command == "ingest":
        result = loop.ingest(args.dataset)
    elif args.command in {"evaluate", "promote"}:
        result = getattr(loop, args.command)(args.sha)
    else:
        result = loop.verify()
    print(json.dumps(result, indent=1, sort_keys=True))
    ok = result.get("promote", result.get("ok", True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
