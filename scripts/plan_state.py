#!/usr/bin/env python3
"""Keep LEGAL_BUILD_PLAN.md's checkboxes and legal_build_state.json in lockstep.

The plan-of-record contract (LEGAL_BUILD_PLAN.md, "State-tracking contract") requires ticking a
checkbox AND updating the state file after every meaningful change. Doing that by hand twice is
how a plan and its state drift apart, so this is the single writer for both files. Never edit
either file by hand.

    python3 scripts/plan_state.py resume                      # START EVERY SESSION WITH THIS
    python3 scripts/plan_state.py section [P3]                # print one phase of the plan
    python3 scripts/plan_state.py tick P1-A [P1-B ...] [--note "..."] [--evidence k=v ...]
    python3 scripts/plan_state.py tick P1-Z --user-confirmed "<the user's exact words>"
    python3 scripts/plan_state.py tick P6-G --evidence job=2601234 log=logs/x.log val_ppl=9.87
    python3 scripts/plan_state.py note "..."
    python3 scripts/plan_state.py next-action "sacct -j 2601234; if COMPLETED do P4-F"
    python3 scripts/plan_state.py block P4-A "USER ACTION: fill scripts/slurm/cluster.env"
    python3 scripts/plan_state.py unblock
    python3 scripts/plan_state.py job add 2601234 --phase P4 --box P4-E --arm setup_env --log logs/setup_env_2601234.log
    python3 scripts/plan_state.py job update 2601234 --status COMPLETED
    python3 scripts/plan_state.py job list
    python3 scripts/plan_state.py artifact set P6.hybrid.s42.last_ckpt /sc/scratch/<user>/lexhybrid/outputs/.../last.ckpt
    python3 scripts/plan_state.py evidence P5 bar_value=0.31
    python3 scripts/plan_state.py verdict P5 "S1 advances: no arm beat it by more than the bar"
    python3 scripts/plan_state.py decision teacher "Qwen/Qwen3-1.7B-Base (P4-U: 8B peak 74 GB > 70)"
    python3 scripts/plan_state.py prereg P5 bar_rule "2x S1 two-seed SD, floor 0.10 PPL"
    python3 scripts/plan_state.py set status "P4 in progress; corpus array running"   # status | scope | helper
    python3 scripts/plan_state.py phase P2 [--status "..."]   # set current_phase explicitly
    python3 scripts/plan_state.py next                        # advance when Pn-Z is ticked
    python3 scripts/plan_state.py show [P2]
    python3 scripts/plan_state.py check                       # gate 0 of scripts/validate.sh
    python3 scripts/plan_state.py init                        # create the state file from the plan
    python3 scripts/plan_state.py sync                        # rebuild state phases from checkboxes
    python3 scripts/plan_state.py readme                      # refresh README.md's progress table

Stdlib only, so it runs with the system python3 before any venv exists.

Grammar the parser understands (LEGAL_BUILD_PLAN.md must follow it exactly):
    ### P3 — Title of the phase                (an em dash, U+2014)
    - [ ] **P3-A** one-line description        (continuation lines are indented and ignored)
    - [x] **P3-Z** the gate box of the phase   (Z is reserved for the gate)
    *Pre-registered prediction ...*            (required in measurement phases)
    **Gate:** ...                              (exactly one per phase)
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PLAN_SETS = {
    "legal": ("LEGAL_BUILD_PLAN.md", "legal_build_state.json"),
}
DEFAULT_PLAN_SET = "legal"

PLAN = ROOT / PLAN_SETS[DEFAULT_PLAN_SET][0]
STATE = ROOT / PLAN_SETS[DEFAULT_PLAN_SET][1]

CHECKBOX_RE = re.compile(r"^(- \[)( |x)(\] \*\*)([A-Z]{1,2}\d+-[A-Z]\d*)(\*\*\s+)(.*)$")
PHASE_RE = re.compile(r"^### ([A-Z]{1,2}\d+)\s+—\s+(.*)$")
GATE_RE = re.compile(r"^\*\*Gate:\*\*")
PREDICTION_RE = re.compile(r"^\*Pre-registered prediction")
USER_ACTION_MARK = "USER ACTION"
MEASUREMENT_WORDS = ("screen", "pretrain", "profil", "eval", "benchmark", "post-training")
OPEN_JOB_STATES = ("SUBMITTED", "PENDING", "RUNNING", "REQUEUED", "UNKNOWN")


def select_plan_set(name: str) -> None:
    global PLAN, STATE
    plan_name, state_name = PLAN_SETS[name]
    PLAN = ROOT / plan_name
    STATE = ROOT / state_name


def _now() -> str:
    return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


def _today() -> str:
    return datetime.date.today().isoformat()


def _clean(s: str, limit: int = 220) -> str:
    """Strip bold markers only. Backticks stay so file paths survive into the state file."""
    s = s.replace("**", "").strip()
    return s[: limit - 3] + "..." if len(s) > limit else s


def _dated(text: str) -> str:
    """Prefix today's date unless the text already starts with an ISO date."""
    if re.match(r"^\d{4}-\d{2}-\d{2}:", text):
        return text
    return f"{_today()}: {text}"


# ----------------------------------------------------------------------------- plan parsing


def parse_plan() -> tuple[list[str], dict[str, dict]]:
    """Return (phase order, {phase_id: {title, checkboxes: {id: {done, desc}}, gates, predictions}})."""
    phases: dict[str, dict] = {}
    order: list[str] = []
    cur: str | None = None
    seen_ids: dict[str, str] = {}
    for line in PLAN.read_text(encoding="utf-8").splitlines():
        m = PHASE_RE.match(line)
        if m:
            cur = m.group(1)
            if cur in phases:
                phases[cur].setdefault("duplicates", []).append(f"phase heading {cur} appears twice")
            else:
                order.append(cur)
                phases[cur] = {
                    "title": _clean(m.group(2)),
                    "checkboxes": {},
                    "gates": 0,
                    "predictions": 0,
                    "duplicates": [],
                }
            continue
        if cur is None:
            continue
        if line.startswith("## "):
            cur = None  # left the phases chapter
            continue
        c = CHECKBOX_RE.match(line)
        if c:
            cid = c.group(4)
            if cid in seen_ids:
                # A dict keyed by id would silently keep only the last one; record it for `check`.
                phases[cur]["duplicates"].append(
                    f"duplicate checkbox id {cid} (in {seen_ids[cid]} and {cur})"
                )
            seen_ids[cid] = cur
            phases[cur]["checkboxes"][cid] = {"done": c.group(2) == "x", "desc": _clean(c.group(6))}
            continue
        if GATE_RE.match(line):
            phases[cur]["gates"] += 1
        if PREDICTION_RE.match(line):
            phases[cur]["predictions"] += 1
    return order, phases


def phase_section(pid: str) -> str:
    """The plan text from `### <pid> —` up to the next `### ` or `## ` heading."""
    lines = PLAN.read_text(encoding="utf-8").splitlines()
    out, inside = [], False
    for line in lines:
        m = PHASE_RE.match(line)
        if m and m.group(1) == pid:
            inside = True
        elif inside and (line.startswith("### ") or line.startswith("## ")):
            break
        if inside:
            out.append(line)
    return "\n".join(out)


def phase_of(box_id: str) -> str:
    return box_id.split("-")[0]


# ----------------------------------------------------------------------------- state io


def _empty_state(order: list[str]) -> dict:
    return {
        "current_phase": order[0] if order else None,
        "last_updated": _now(),
        "status": "created by plan_state.py init",
        "plan_of_record": PLAN.name,
        "helper": "python3 scripts/plan_state.py ...",
        "scope": "",
        "blocked_on": None,
        "next_action": "",
        "decisions": {},
        "pre_registered": {},
        "phase_order": [],
        "phases": {},
        "jobs": {},
        "artifacts": {},
        "sessions": [],
        "notes": [],
        "open_questions": [],
    }


def load_state(create_if_missing: bool = False) -> dict:
    if not STATE.exists():
        if not create_if_missing:
            print(
                f"ERROR: {STATE.name} not found. Run `python3 scripts/plan_state.py init` first.",
                file=sys.stderr,
            )
            sys.exit(1)
        order, _ = parse_plan()
        return refresh_phases(_empty_state(order))
    return json.loads(STATE.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    state["last_updated"] = _now()
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, STATE)


def set_checkboxes(ids: list[str], done: bool) -> list[str]:
    """Flip checkboxes in the plan markdown. Returns the ids actually changed."""
    text = PLAN.read_text(encoding="utf-8")
    out, changed = [], []
    for line in text.splitlines(keepends=True):
        c = CHECKBOX_RE.match(line.rstrip("\n"))
        if c and c.group(4) in ids:
            if (c.group(2) == "x") != done:
                changed.append(c.group(4))
            line = f"{c.group(1)}{'x' if done else ' '}{c.group(3)}{c.group(4)}{c.group(5)}{c.group(6)}\n"
        out.append(line)
    tmp = PLAN.with_suffix(".md.tmp")
    tmp.write_text("".join(out), encoding="utf-8")
    os.replace(tmp, PLAN)
    return changed


def refresh_phases(state: dict) -> dict:
    """Rebuild state['phases'] from the plan, preserving evidence/verdict."""
    order, phases = parse_plan()
    prev = state.get("phases", {})
    merged = {}
    for pid in order:
        old = prev.get(pid, {})
        boxes = phases[pid]["checkboxes"]
        n_done = sum(1 for b in boxes.values() if b["done"])
        if n_done == 0:
            status = "pending"
        elif n_done == len(boxes):
            status = "complete"
        else:
            status = "in_progress"
        merged[pid] = {
            "title": phases[pid]["title"],
            "status": status,
            "checkboxes": boxes,
            "evidence": old.get("evidence", {}),
            "verdict": old.get("verdict"),
        }
    state["phase_order"] = order
    state["phases"] = merged
    return state


def _current_session(state: dict) -> dict:
    sessions = state.setdefault("sessions", [])
    if not sessions or sessions[-1].get("ended"):
        sessions.append({"started": _now(), "ended": None, "ticked": [], "jobs": []})
    return sessions[-1]


# ----------------------------------------------------------------------------- commands


def cmd_init(args, state_unused=None) -> int:
    if STATE.exists() and not args.force:
        print(f"ERROR: {STATE.name} exists; refusing to overwrite (use --force).", file=sys.stderr)
        return 1
    order, _ = parse_plan()
    if not order:
        print(f"ERROR: no `### Pn — title` phases found in {PLAN.name}", file=sys.stderr)
        return 1
    state = refresh_phases(_empty_state(order))
    state["notes"].append(_dated(f"state file created from {PLAN.name} ({len(order)} phases)"))
    save_state(state)
    print(f"created {STATE.name}: phases {', '.join(order)}; current_phase = {state['current_phase']}")
    return 0


def cmd_sync(args, state) -> int:
    state = refresh_phases(state)
    save_state(state)
    print(f"state phases regenerated from {PLAN.name} checkboxes")
    return 0


def cmd_tick(args, state) -> int:
    order, plan_phases = parse_plan()
    known = {cid: ph["checkboxes"][cid] for ph in plan_phases.values() for cid in ph["checkboxes"]}
    unknown = [i for i in args.ids if i not in known]
    if unknown:
        print(f"ERROR: unknown checkbox id(s): {', '.join(unknown)}", file=sys.stderr)
        return 1
    if not args.undo:
        job_bound = {j.get("box") for j in state.get("jobs", {}).values()}
        for cid in args.ids:
            pid = phase_of(cid)
            if pid != state.get("current_phase") and not args.force and cid not in job_bound:
                print(
                    f"ERROR: {cid} is in phase {pid} but current_phase is {state.get('current_phase')}. "
                    f"Use `phase {pid}` / `next` first, or --force if this is deliberate.",
                    file=sys.stderr,
                )
                return 1
            desc = known[cid]["desc"]
            if USER_ACTION_MARK in desc and not args.user_confirmed:
                print(
                    f"ERROR: {cid} is a USER ACTION box. It may only be ticked with "
                    f'--user-confirmed "<the user\'s exact words>" after the user acted.',
                    file=sys.stderr,
                )
                return 1
            if "sbatch" in desc and not args.evidence:
                print(
                    f"ERROR: {cid} submits or reads a cluster job; pass --evidence job=<id> "
                    f"(and log=<path>, metrics) so the record carries the job id.",
                    file=sys.stderr,
                )
                return 1
    for item in args.evidence:
        if "=" not in item:
            print(f"ERROR: --evidence expects KEY=VALUE, got {item!r}", file=sys.stderr)
            return 1
    changed = set_checkboxes(args.ids, done=not args.undo)
    state = refresh_phases(state)
    pid = phase_of(args.ids[0])
    for item in args.evidence:
        k, v = item.split("=", 1)
        state["phases"][pid]["evidence"][k] = v
    if args.user_confirmed:
        for cid in args.ids:
            state["phases"][phase_of(cid)]["evidence"][f"{cid}_user_confirmed"] = args.user_confirmed
    verb = "unticked" if args.undo else "ticked"
    state.setdefault("notes", []).append(
        _dated(args.note) if args.note else _dated(f"{verb} {', '.join(args.ids)}")
    )
    if not args.undo:
        _current_session(state)["ticked"].extend(args.ids)
    save_state(state)
    noop = set(args.ids) - set(changed)
    print(f"{verb}: {', '.join(args.ids)}" + (f"  (no-op for {sorted(noop)})" if noop else ""))
    return 0


def cmd_note(args, state) -> int:
    state.setdefault("notes", []).append(_dated(args.text))
    save_state(state)
    print("note appended")
    return 0


def cmd_next_action(args, state) -> int:
    state["next_action"] = args.text
    save_state(state)
    print(f"next_action = {args.text}")
    return 0


def cmd_phase(args, state) -> int:
    if args.name not in state.get("phase_order", []):
        print(
            f"ERROR: {args.name} is not a phase; known: {', '.join(state.get('phase_order', []))}",
            file=sys.stderr,
        )
        return 1
    state["current_phase"] = args.name
    if args.status:
        state["status"] = args.status
    state.setdefault("notes", []).append(_dated(f"current_phase -> {args.name}"))
    save_state(state)
    print(f"current_phase = {args.name}")
    return 0


def cmd_next(args, state) -> int:
    state = refresh_phases(state)
    cur = state["current_phase"]
    order = state["phase_order"]
    gate = f"{cur}-Z"
    boxes = state["phases"][cur]["checkboxes"]
    if gate not in boxes or not boxes[gate]["done"]:
        print(f"ERROR: gate box {gate} is not ticked; the phase is not complete.", file=sys.stderr)
        return 1
    idx = order.index(cur)
    if idx + 1 >= len(order):
        print("already at the last phase")
        return 0
    state["current_phase"] = order[idx + 1]
    state["blocked_on"] = None
    state.setdefault("notes", []).append(_dated(f"{cur} complete -> current_phase {order[idx + 1]}"))
    save_state(state)
    print(f"current_phase = {state['current_phase']}")
    return 0


def cmd_block(args, state) -> int:
    state["blocked_on"] = {"box": args.box, "what": args.what, "since": _today()}
    state.setdefault("notes", []).append(_dated(f"BLOCKED on {args.box}: {args.what}"))
    save_state(state)
    print(f"blocked_on = {args.box}: {args.what}")
    return 0


def cmd_unblock(args, state) -> int:
    prev = state.get("blocked_on")
    state["blocked_on"] = None
    state.setdefault("notes", []).append(_dated(f"unblocked ({prev['box'] if prev else 'nothing'})"))
    save_state(state)
    print("blocked_on = null")
    return 0


def cmd_job(args, state) -> int:
    jobs = state.setdefault("jobs", {})
    if args.job_cmd == "add":
        jobs[args.id] = {
            "phase": args.phase,
            "box": args.box,
            "arm": args.arm,
            "submitted": _now(),
            "status": "SUBMITTED",
            "log": args.log,
            "last_checked": None,
        }
        state.setdefault("notes", []).append(
            _dated(f"submitted job {args.id} ({args.box} {args.arm or ''})".rstrip())
        )
        _current_session(state)["jobs"].append(args.id)
        save_state(state)
        print(f"job {args.id} recorded for {args.box}")
        return 0
    if args.job_cmd == "update":
        if args.id not in jobs:
            print(f"ERROR: unknown job {args.id}", file=sys.stderr)
            return 1
        jobs[args.id]["status"] = args.status
        jobs[args.id]["last_checked"] = _now()
        if args.note:
            jobs[args.id]["note"] = args.note
        state.setdefault("notes", []).append(_dated(f"job {args.id} -> {args.status}"))
        save_state(state)
        print(f"job {args.id} = {args.status}")
        return 0
    if args.job_cmd == "list":
        if not jobs:
            print("no jobs recorded")
        for jid, j in jobs.items():
            print(
                f"{jid:<10} {j.get('status', '?'):<10} {j.get('phase', '')}/{j.get('box', '')} "
                f"{j.get('arm') or '':<14} {j.get('log') or ''}"
            )
        return 0
    return 1


def cmd_artifact(args, state) -> int:
    state.setdefault("artifacts", {})[args.key] = args.value
    save_state(state)
    print(f"artifact {args.key} = {args.value}")
    return 0


def cmd_evidence(args, state) -> int:
    if args.phase not in state.get("phases", {}):
        print(f"ERROR: unknown phase {args.phase}", file=sys.stderr)
        return 1
    for item in args.items:
        if "=" not in item:
            print(f"ERROR: expects KEY=VALUE, got {item!r}", file=sys.stderr)
            return 1
        k, v = item.split("=", 1)
        state["phases"][args.phase]["evidence"][k] = v
    save_state(state)
    print(f"evidence recorded under {args.phase}: {', '.join(args.items)}")
    return 0


def cmd_verdict(args, state) -> int:
    if args.phase not in state.get("phases", {}):
        print(f"ERROR: unknown phase {args.phase}", file=sys.stderr)
        return 1
    state["phases"][args.phase]["verdict"] = args.text
    state.setdefault("notes", []).append(_dated(f"verdict {args.phase}: {args.text}"))
    save_state(state)
    print(f"verdict {args.phase} recorded")
    return 0


def cmd_decision(args, state) -> int:
    state.setdefault("decisions", {})[args.key] = args.value
    state.setdefault("notes", []).append(_dated(f"decision {args.key}: {args.value}"))
    save_state(state)
    print(f"decisions.{args.key} = {args.value}")
    return 0


def cmd_prereg(args, state) -> int:
    pre = state.setdefault("pre_registered", {})
    if args.phase not in state.get("phase_order", []) and args.phase != "rules":
        print(f"ERROR: {args.phase} is not a phase (or 'rules')", file=sys.stderr)
        return 1
    entry = pre.setdefault(args.phase, {})
    entry[args.key] = args.value
    if "written" not in entry:
        entry["written"] = _now()
    state.setdefault("notes", []).append(_dated(f"pre-registered {args.phase}.{args.key}"))
    save_state(state)
    print(f"pre_registered.{args.phase}.{args.key} recorded")
    return 0


SETTABLE = ("status", "scope", "helper", "next_action")


def cmd_set(args, state) -> int:
    if args.key not in SETTABLE:
        print(
            f"ERROR: `set` only writes {SETTABLE}; use decision/prereg/evidence/verdict for the rest",
            file=sys.stderr,
        )
        return 1
    state[args.key] = args.value
    save_state(state)
    print(f"{args.key} = {args.value}")
    return 0


def cmd_show(args, state) -> int:
    state = refresh_phases(state)
    print(f"current_phase : {state['current_phase']}")
    print(f"status        : {state.get('status', '')}")
    print(f"last_updated  : {state.get('last_updated', '')}\n")
    for pid in state["phase_order"]:
        ph = state["phases"][pid]
        if args.phase and pid != args.phase:
            continue
        boxes = ph["checkboxes"]
        done = sum(1 for b in boxes.values() if b["done"])
        mark = {"complete": "x", "in_progress": "~", "pending": " "}[ph["status"]]
        print(f"[{mark}] {pid:<4} {done}/{len(boxes):<3} {ph['title']}")
        if args.phase:
            for cid, b in boxes.items():
                print(f"      [{'x' if b['done'] else ' '}] {cid}  {b['desc']}")
    return 0


def cmd_section(args, state) -> int:
    pid = args.phase or state.get("current_phase")
    text = phase_section(pid)
    if not text:
        print(f"ERROR: no section for {pid}", file=sys.stderr)
        return 1
    print(text)
    return 0


def cmd_resume(args, state) -> int:
    state = refresh_phases(state)
    cur = state["current_phase"]
    ph = state["phases"][cur]
    done = sum(1 for b in ph["checkboxes"].values() if b["done"])
    print(f"=== {PLAN.name} — resume ===")
    print(f"current_phase : {cur}  ({ph['title']})  {done}/{len(ph['checkboxes'])} ticked")
    print(f"status        : {state.get('status', '')}")
    b = state.get("blocked_on")
    print("blocked_on    : " + (f"{b['box']} — {b['what']} (since {b['since']})" if b else "none"))
    print(f"next_action   : {state.get('next_action') or '(none set)'}")
    open_jobs = {k: v for k, v in state.get("jobs", {}).items() if v.get("status") in OPEN_JOB_STATES}
    print("open jobs     : " + ("none" if not open_jobs else ""))
    for jid, j in open_jobs.items():
        print(
            f"    {jid}  {j.get('status')}  {j.get('phase')}/{j.get('box')}  {j.get('arm') or ''}  {j.get('log') or ''}"
        )
    print("last notes    :")
    for n in state.get("notes", [])[-5:]:
        print(f"    - {n}")
    _current_session(state)
    save_state(state)
    print(f"\n--- {cur} section of {PLAN.name} (read nothing else of the plan) ---\n")
    print(phase_section(cur))
    return 0


def cmd_session_end(args, state) -> int:
    sessions = state.setdefault("sessions", [])
    if sessions and not sessions[-1].get("ended"):
        sessions[-1]["ended"] = _now()
    if args.summary:
        state.setdefault("notes", []).append(_dated(f"session end: {args.summary}"))
    save_state(state)
    print("session closed")
    return 0


def cmd_check(args, state) -> int:
    """Structural invariants of plan + state. Exit 1 on the first class of failure found."""
    problems: list[str] = []
    order, phases = parse_plan()
    text = PLAN.read_text(encoding="utf-8")

    # 1. unique ids and headings, and id prefix matches its heading
    for pid, ph in phases.items():
        problems.extend(ph.get("duplicates", []))
        for cid in ph["checkboxes"]:
            if phase_of(cid) != pid:
                problems.append(f"checkbox {cid} sits under phase {pid}")
    # 2. gates and Z boxes
    for pid, ph in phases.items():
        if ph["gates"] != 1:
            problems.append(f"phase {pid} has {ph['gates']} '**Gate:**' lines (need exactly 1)")
        if f"{pid}-Z" not in ph["checkboxes"]:
            problems.append(f"phase {pid} has no {pid}-Z gate checkbox")
        title = ph["title"].lower()
        if any(w in title for w in MEASUREMENT_WORDS) and ph["predictions"] == 0:
            problems.append(
                f"measurement phase {pid} ('{ph['title']}') has no '*Pre-registered prediction' line"
            )
    # 3. state consistency
    if state.get("current_phase") not in order:
        problems.append(f"current_phase {state.get('current_phase')!r} is not in the plan's phases {order}")
    if STATE.exists():
        disk = json.loads(STATE.read_text(encoding="utf-8"))
        n_md = sum(len(ph["checkboxes"]) for ph in phases.values())
        n_json = sum(len(ph.get("checkboxes", {})) for ph in disk.get("phases", {}).values())
        if n_md != n_json:
            problems.append(f"checkbox count differs: plan {n_md} vs state {n_json} (run `sync`)")
        for pid, ph in disk.get("phases", {}).items():
            for cid, box in ph.get("checkboxes", {}).items():
                plan_box = phases.get(pid, {}).get("checkboxes", {}).get(cid)
                if plan_box is not None and plan_box["done"] != box.get("done"):
                    problems.append(
                        f"{cid}: plan says done={plan_box['done']} but state says {box.get('done')} (run `sync`)"
                    )
                if box.get("done") and USER_ACTION_MARK in box.get("desc", ""):
                    if f"{cid}_user_confirmed" not in ph.get("evidence", {}):
                        problems.append(
                            f"{cid} is a ticked USER ACTION box without a recorded user confirmation"
                        )
        notes = " ".join(disk.get("notes", []))
        for pid, ph in disk.get("phases", {}).items():
            if pid in ("P0", "P1"):
                continue
            for cid, box in ph.get("checkboxes", {}).items():
                if box.get("done") and cid not in notes and not ph.get("evidence"):
                    problems.append(f"{cid} is ticked but no note or evidence mentions it")
    # 4. hygiene: no bare `python scripts/` lines meant for the login node
    for i, line in enumerate(text.splitlines(), 1):
        if re.match(r"^\s*python scripts/", line):
            problems.append(
                f"line {i}: bare 'python scripts/...' (write python3 scripts/... or an sbatch wrapper)"
            )

    if problems:
        print("CHECK FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    n = sum(len(ph["checkboxes"]) for ph in phases.values())
    print(f"OK: {len(order)} phases, {n} checkboxes, gates and predictions present, state consistent")
    return 0


def cmd_readme(args, state) -> int:
    state = refresh_phases(state)
    save_state(state)
    readme = ROOT / "README.md"
    if not readme.exists():
        print("ERROR: README.md not found", file=sys.stderr)
        return 1
    text = readme.read_text(encoding="utf-8")
    done = sum(1 for p in state["phases"].values() for c in p["checkboxes"].values() if c["done"])
    total = sum(len(p["checkboxes"]) for p in state["phases"].values())
    complete = [pid for pid in state["phase_order"] if state["phases"][pid]["status"] == "complete"]
    span = f"{complete[0]}–{complete[-1]}" if len(complete) > 1 else (complete[0] if complete else "none")
    old_status = re.search(r"\*\*Status: .*?\*\*", text)
    if old_status:
        text = text.replace(
            old_status.group(0),
            f"**Status: {span} complete, {done}/{total} checkboxes; current phase {state['current_phase']}.**",
            1,
        )
    rows = ["| Phase | Title | Status |", "|---|---|---|"]
    for pid in state["phase_order"]:
        ph = state["phases"][pid]
        n = sum(1 for c in ph["checkboxes"].values() if c["done"])
        mark = {"complete": "✅", "in_progress": "\U0001f504", "pending": "⬜"}[ph["status"]]
        rows.append(f"| {pid} | {ph['title']} | {mark} {n}/{len(ph['checkboxes'])} |")
    table = "\n".join(rows)
    if "| Phase | Title | Status |" in text:
        start = text.index("| Phase | Title | Status |")
        end = text.find("\n\n", start)
        text = text[:start] + table + ("\n" if end == -1 else text[end:])
    else:
        text = text.rstrip() + "\n\n## Progress\n\n" + table + "\n"
    readme.write_text(text, encoding="utf-8")
    print(f"README refreshed: {span} complete, {done}/{total} checkboxes")
    return 0


# ----------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", choices=sorted(PLAN_SETS), default=DEFAULT_PLAN_SET)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create the state file from the plan")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init, needs_state=False)

    sub.add_parser("sync", help="regenerate state phases from plan checkboxes").set_defaults(
        func=cmd_sync, create=True
    )

    p = sub.add_parser("resume", help="print everything a new session needs")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("section", help="print one phase of the plan")
    p.add_argument("phase", nargs="?", default=None)
    p.set_defaults(func=cmd_section)

    p = sub.add_parser("tick", help="mark checkbox(es) done in plan + state")
    p.add_argument("ids", nargs="+")
    p.add_argument("--note", default=None)
    p.add_argument("--evidence", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--undo", action="store_true")
    p.add_argument("--force", action="store_true", help="tick a box outside current_phase")
    p.add_argument(
        "--user-confirmed",
        default=None,
        metavar="QUOTE",
        help="required for USER ACTION boxes: the user's exact words",
    )
    p.set_defaults(func=cmd_tick)

    p = sub.add_parser("note", help="append a dated one-line note")
    p.add_argument("text")
    p.set_defaults(func=cmd_note)

    p = sub.add_parser("next-action", help="set the one-line next_action")
    p.add_argument("text")
    p.set_defaults(func=cmd_next_action)

    p = sub.add_parser("phase", help="set current_phase explicitly")
    p.add_argument("name")
    p.add_argument("--status", default=None)
    p.set_defaults(func=cmd_phase)

    sub.add_parser("next", help="advance to the next phase (requires Pn-Z ticked)").set_defaults(
        func=cmd_next
    )

    p = sub.add_parser("block", help="record that work is blocked on a box (usually a USER ACTION)")
    p.add_argument("box")
    p.add_argument("what")
    p.set_defaults(func=cmd_block)

    sub.add_parser("unblock", help="clear blocked_on").set_defaults(func=cmd_unblock)

    p = sub.add_parser("job", help="track cluster jobs")
    js = p.add_subparsers(dest="job_cmd", required=True)
    a = js.add_parser("add")
    a.add_argument("id")
    a.add_argument("--phase", required=True)
    a.add_argument("--box", required=True)
    a.add_argument("--arm", default=None)
    a.add_argument("--log", default=None)
    u = js.add_parser("update")
    u.add_argument("id")
    u.add_argument("--status", required=True)
    u.add_argument("--note", default=None)
    js.add_parser("list")
    p.set_defaults(func=cmd_job)

    p = sub.add_parser("artifact", help="record a cluster-side artifact path")
    asub = p.add_subparsers(dest="art_cmd", required=True)
    s = asub.add_parser("set")
    s.add_argument("key")
    s.add_argument("value")
    p.set_defaults(func=cmd_artifact)

    p = sub.add_parser("evidence", help="record KEY=VALUE evidence under a phase")
    p.add_argument("phase")
    p.add_argument("items", nargs="+", metavar="KEY=VALUE")
    p.set_defaults(func=cmd_evidence)

    p = sub.add_parser("verdict", help="record a phase verdict")
    p.add_argument("phase")
    p.add_argument("text")
    p.set_defaults(func=cmd_verdict)

    p = sub.add_parser("decision", help="record a design decision (decisions.<key>)")
    p.add_argument("key")
    p.add_argument("value")
    p.set_defaults(func=cmd_decision)

    p = sub.add_parser("prereg", help="record a pre-registered prediction/bar (pre_registered.<phase>.<key>)")
    p.add_argument("phase")
    p.add_argument("key")
    p.add_argument("value")
    p.set_defaults(func=cmd_prereg)

    p = sub.add_parser("set", help="set status | scope | helper | next_action")
    p.add_argument("key")
    p.add_argument("value")
    p.set_defaults(func=cmd_set)

    p = sub.add_parser("show", help="show progress")
    p.add_argument("phase", nargs="?", default=None)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("session-end", help="close the current session entry")
    p.add_argument("--summary", default=None)
    p.set_defaults(func=cmd_session_end)

    sub.add_parser("check", help="validate plan + state invariants (exit 1 on failure)").set_defaults(
        func=cmd_check, create=True
    )
    sub.add_parser("readme", help="refresh README.md's progress table").set_defaults(func=cmd_readme)

    args = ap.parse_args(argv)
    select_plan_set(args.plan)
    if not PLAN.exists():
        print(f"ERROR: {PLAN} not found", file=sys.stderr)
        return 1
    if getattr(args, "needs_state", True) is False:
        return args.func(args)
    state = load_state(create_if_missing=getattr(args, "create", False))
    return args.func(args, state)


if __name__ == "__main__":
    sys.exit(main())
