#!/usr/bin/env python3
"""MCP server for ad hoc QA of the duct-noise pipeline by another Claude session.

Stdio JSON-RPC, standard library only. Every call builds the current code
first (a no-op when nothing changed), so the QA session always tests what
the dev session last saved, without reconnecting. Reports are cached under
samples/.qa-cache (gitignored, as client models are confidential).
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLES = Path(os.environ.get("DUCT_SAMPLES", REPO / "samples"))
CACHE = SAMPLES / ".qa-cache"
PIPELINE = REPO / "rvt-read" / "wasm"
ANALYSE = PIPELINE / "target" / "release" / "examples" / "analyse"
ENGINE = REPO / "rust"
ENV = {**os.environ, "DUCT_SAMPLES": str(SAMPLES), "RVT_NORDIC_LCA_DIR": str(SAMPLES)}

INSTRUCTIONS = """\
You are QA for duct-inverse: it reads a Revit (.rvt) and/or IFC model of an
HVAC duct system, heals the duct network's connectivity, finds each air
terminal's path back to its unit (AHU, fan coil, heat pump), and predicts
the noise each room gets (octave bands 63 Hz-8 kHz, NC rating) with an
empirical acoustic engine (ASHRAE / CIBSE Guide B / TM 5-805-4 methods).
Where it cannot handle a model automatically it should say so in the
report's `issues` (kind, severity, message, check, elements) for the
engineer, and what it assumed in `assumptions`.

Look for: results that are physically implausible (an NC far out of line
with flow, size and room; attenuation that grows the wrong way; velocities
that do not follow from flow and size); counts that do not add up; rooms,
terminals or units the report silently drops; issues whose message is
wrong, unclear, duplicated or names the wrong elements; disagreement
between a model's .rvt and its IFC export; crashes, slowness, wording an
engineer would misread. Revit's internal units are feet and ft^3/s; the
report is SI (m, L/s, dB).

Tools: status (commit under test), list_models, analyse (summary of a run),
report (any part of the full report), find_element (everything the report
says about one element id), engine (rate a hand-built chain directly, for
checking the physics against handbook values and its trends), run_tests.
Report each finding with the commit, model, what you saw, what you
expected and why, and the element ids.
"""


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def build(cwd, *args):
    """Build, or raise the compiler's complaint."""
    r = subprocess.run(["cargo", "build", "-q", *args], cwd=cwd, env=ENV, capture_output=True, text=True)
    if r.returncode:
        raise ToolError("build failed:\n" + r.stderr[-6000:])


class ToolError(Exception):
    pass


def model_path(name):
    if not name:
        return None
    p = Path(name).expanduser()
    if not p.is_absolute():
        p = SAMPLES / name
    if not p.exists():
        raise ToolError(f"no model {p}; list_models shows those in {SAMPLES}")
    return p


def run(args):
    """The full report for a model under the current code, cached."""
    model, ifc = model_path(args.get("model")), model_path(args.get("ifc"))
    if model is None:
        raise ToolError("model is required")
    if model.suffix.lower() == ".ifc" and ifc is None:
        model, ifc = None, model
    build(PIPELINE, "--release", "--example", "analyse")
    lw, bend = args.get("lw"), args.get("bend_regen")
    key = json.dumps([[str(p), p.stat().st_size, p.stat().st_mtime] for p in (model, ifc) if p]
                     + [lw, bend, ANALYSE.stat().st_mtime])
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if cached.exists() and not args.get("fresh"):
        # The same binary and model: the result holds at the current commit.
        report = json.loads(cached.read_text())
        report["_run"].update(commit=commit(), cache_hit=True, first_run_at=report["_run"]["commit"])
        return report
    env = dict(ENV)
    if lw:
        env["LW"] = ",".join(str(v) for v in lw)
    if bend:
        env["BEND_REGEN"] = bend
    cmd = [str(ANALYSE)] + [str(p) for p in (model, ifc) if p]
    start = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        raise ToolError(f"analyse failed ({r.returncode}):\n{r.stderr[-6000:]}")
    report = {"_run": {"commit": commit(), "cache_hit": False, "models": [p.name for p in (model, ifc) if p],
                       "lw": lw, "bend_regen": bend or "recorded", "wall_s": round(time.time() - start, 1),
                       "stderr": r.stderr[-2000:]},
              **json.loads(r.stdout)}
    cached.write_text(json.dumps(report))
    return report


def commit():
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return git("rev-parse", "--short", "HEAD") + (" +uncommitted" if dirty else "")


def summary(report):
    """The report less its bulk: per-terminal rows become counts, the worst
    path's blocks one line each."""
    out = dict(report)
    terms = report.get("terminals") or []
    ncs = sorted(t["room_nc"] for t in terms if t.get("room_nc") is not None)
    out["terminals"] = {"count": len(terms), "rated": len(ncs),
                        "room_nc_min_median_max": [ncs[0], ncs[len(ncs) // 2], ncs[-1]] if ncs else None,
                        "detail": "report(path='terminals')"}
    path = report.get("path")
    if isinstance(path, dict) and isinstance(path.get("blocks"), list):
        path = dict(path)
        path["blocks"] = [f"{b.get('index')}: {b.get('type')} el {b.get('element')} {b.get('note', '')} "
                          f"v={b.get('velocity_m_s')} lw={b.get('lw_db')}" for b in path["blocks"]]
        out["path"] = path
    return out


def select(value, path):
    for part in [p for p in re.split(r"[.\[\]]", path or "") if p]:
        if isinstance(value, list):
            value = value[int(part)]
        elif isinstance(value, dict):
            if part not in value:
                raise ToolError(f"no key {part!r}; have {list(value)}")
            value = value[part]
        else:
            raise ToolError(f"cannot index {type(value).__name__} with {part!r}")
    return value


def matches(row, where):
    for cond in where or []:
        m = re.fullmatch(r"\s*([\w.]+)\s*(=|!=|>=|<=|>|<|~)\s*(.*?)\s*", cond)
        if not m:
            raise ToolError(f"bad condition {cond!r}: use field=value, != > < >= <= or ~ (substring)")
        field, op, want = m.groups()
        try:
            have = select(row, field)
        except (ToolError, IndexError, ValueError):
            return False
        if op == "~":
            if want.lower() not in json.dumps(have).lower():
                return False
            continue
        try:
            want_v = json.loads(want)
        except json.JSONDecodeError:
            want_v = want
        try:
            ok = {"=": have == want_v, "!=": have != want_v, ">": have > want_v, "<": have < want_v,
                  ">=": have >= want_v, "<=": have <= want_v}[op]
        except TypeError:
            ok = False
        if not ok:
            return False
    return True


def tool_report(args):
    value = select(run(args), args.get("path"))
    if isinstance(value, list):
        rows = [r for r in value if matches(r, args.get("where"))]
        if args.get("fields"):
            rows = [{f: select(r, f) if isinstance(r, dict) and f.split(".")[0] in r else None
                     for f in args["fields"]} for r in rows]
        offset, limit = args.get("offset", 0), args.get("limit", 50)
        return {"total": len(value), "matched": len(rows), "offset": offset, "rows": rows[offset:offset + limit]}
    return value


def tool_find_element(args):
    target = int(args["id"])
    hits = []

    def walk(v, path, parent):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{path}.{k}" if path else k, v)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{path}[{i}]", v)
        elif v == target and not isinstance(v, bool):
            ctx = json.dumps(parent)
            hits.append({"at": path, "in": parent if len(ctx) < 1500 else ctx[:1500] + "…"})

    walk(run(args), "", None)
    return {"id": target, "found": len(hits), "hits": hits[:40]}


def tool_status(_):
    return {"commit": commit(), "subject": git("log", "-1", "--format=%s"),
            "recent": git("log", "-8", "--format=%h %s").splitlines(),
            "uncommitted": git("status", "--porcelain", "--untracked-files=no").splitlines(),
            "samples": str(SAMPLES)}


def tool_list_models(_):
    files = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in (".rvt", ".ifc"))
    stems = {}
    for p in files:
        stems.setdefault(p.stem, []).append(p.suffix.lower())
    return {"dir": str(SAMPLES),
            "models": [{"file": p.name, "mb": round(p.stat().st_size / 1e6, 1),
                        "pair": f"{p.stem}.ifc" if p.suffix.lower() == ".rvt" and ".ifc" in stems[p.stem] else None}
                       for p in files],
            "note": "analyse a .rvt alone, an .ifc alone, or a .rvt with its IFC export as ifc; "
                    "any other path is taken as given"}


def tool_engine(args):
    r = subprocess.run(["cargo", "run", "-q", "-p", "wall-duct-wasm", "--example", "evaluate"], cwd=ENGINE,
                       env=ENV, input=json.dumps(args["chain"]), capture_output=True, text=True)
    if r.returncode:
        raise ToolError(r.stderr[-6000:])
    return json.loads(r.stdout)


SUITES = {
    "engine": (ENGINE, ["cargo", "test", "--release", "-p", "wall-duct-core", "-p", "wall-duct-wasm"]),
    "heal": (REPO / "rvt-read", ["cargo", "test", "--release", "--features", "rvt"]),
    "pipeline": (PIPELINE, ["cargo", "test", "--release"]),
}


def tool_run_tests(args):
    out = {}
    for name in [args.get("suite", "all")] if args.get("suite", "all") != "all" else SUITES:
        cwd, cmd = SUITES[name]
        r = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True)
        text = r.stdout + r.stderr
        lines = [l for l in text.splitlines() if l.startswith("test result") or "FAILED" in l or "panicked" in l]
        out[name] = {"passed": r.returncode == 0, "lines": lines[-60:],
                     **({} if r.returncode == 0 else {"tail": text[-4000:]})}
    return out


MODEL_ARGS = {
    "model": {"type": "string", "description": "file in the samples dir (see list_models) or an absolute path; .rvt or .ifc"},
    "ifc": {"type": "string", "description": "the IFC export to read with a .rvt model"},
    "lw": {"type": "array", "items": {"type": "number"}, "description": "unit sound power, 8 octave bands 63 Hz-8 kHz, dB; default the engine's"},
    "bend_regen": {"type": "string", "enum": ["recorded", "cibse", "woods"], "description": "how bends' regenerated noise is found"},
    "fresh": {"type": "boolean", "description": "ignore the cache and rerun"},
}

TOOLS = {
    "status": (tool_status, "The commit under test, recent commits and uncommitted changes.", {}, []),
    "list_models": (tool_list_models, "Models available to analyse, and which .rvt has its IFC export alongside.", {}, []),
    "analyse": (lambda a: summary(run(a)),
                "Run the pipeline on a model under the current code (builds first; 5-60 s, cached until the code or "
                "model changes) and summarise the report: counts, rooms and their NC, issues, assumptions, warnings, "
                "the worst room's path block by block, timings.", MODEL_ARGS, ["model"]),
    "report": (tool_report,
               "Any part of a run's full report. `path` is dotted (rooms, terminals, issues[2].elements, path.blocks). "
               "On a list, `where` filters rows (e.g. [\"room_nc>=45\", \"space~studio\"]), `fields` projects, "
               "limit/offset page (default 50).",
               {**MODEL_ARGS, "path": {"type": "string"}, "where": {"type": "array", "items": {"type": "string"}},
                "fields": {"type": "array", "items": {"type": "string"}}, "limit": {"type": "integer"},
                "offset": {"type": "integer"}}, ["model"]),
    "find_element": (tool_find_element, "Everywhere a Revit/IFC element id appears in a run's report, with the object around it.",
                     {**MODEL_ARGS, "id": {"type": "integer"}}, ["model", "id"]),
    "engine": (tool_engine,
               "Rate a hand-built supply chain directly with the acoustic engine (the web app's evaluate), for checking "
               "physics against handbook values and trends. chain = {source_type: fcu|vav|axial_fan, source_offset_db, "
               "volume_flow_m3s, room_volume_m3, receiver_distance_m, lined_fit?: reynolds_bevirt|ari885, blocks: [ "
               "{kind: segment, length_m, width_m, height_m, lining_m} | {kind: bend, width_m, height_m} | "
               "{kind: attenuator, length_m, width_m, height_m, free_area_ratio, splitter_m} | "
               "{kind: diffuser, width_m, height_m} ]}. Returns NC, receiver SPL per band against the NC curve, "
               "block labels and sensitivities. The source types are fixed catalogue spectra shifted by "
               "source_offset_db: they do not follow volume_flow (flow reaches the result through regenerated "
               "noise only), unlike the pipeline, which estimates each unit's fan from its flow. Inputs that make no "
               "physical sense (zero flow, a lining that fills the duct) are refused.",
               {"chain": {"type": "object"}}, ["chain"]),
    "run_tests": (tool_run_tests, "Run a test suite: engine (acoustic engine), heal (network reading), pipeline "
                  "(whole pipeline on the samples), or all. Minutes, not seconds.",
                  {"suite": {"type": "string", "enum": ["engine", "heal", "pipeline", "all"]}}, []),
}


def handle(msg):
    method, params = msg.get("method"), msg.get("params") or {}
    if method == "initialize":
        return {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "duct-qa", "version": "0.1"},
                "instructions": INSTRUCTIONS}
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": [{"name": n, "description": d,
                           "inputSchema": {"type": "object", "properties": props, "required": req}}
                          for n, (_, d, props, req) in TOOLS.items()]}
    if method == "tools/call":
        tool = TOOLS.get(str(params.get("name")))
        if tool is None:
            return {"content": [{"type": "text", "text": f"no tool {params.get('name')}"}], "isError": True}
        try:
            result = tool[0](params.get("arguments") or {})
            return {"content": [{"type": "text", "text": json.dumps(result, indent=1)}]}
        except ToolError as e:
            return {"content": [{"type": "text", "text": str(e)}], "isError": True}
        except Exception as e:  # a bug here, not in the code under test
            return {"content": [{"type": "text", "text": f"qa server error: {type(e).__name__}: {e}"}], "isError": True}
    raise LookupError(method)


def reply_to(msg):
    """The JSON-RPC reply to one message, or None for a notification."""
    if "id" not in msg:
        return None
    try:
        return {"jsonrpc": "2.0", "id": msg["id"], "result": handle(msg)}
    except LookupError as e:
        return {"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"no method {e}"}}


def serve_stdio():
    # This file changed under a running session: become the new one after
    # answering (stateless, so the client never notices).
    born = Path(__file__).stat().st_mtime
    for line in sys.stdin:
        if line.strip() and (reply := reply_to(json.loads(line))):
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
        if Path(__file__).stat().st_mtime != born:
            os.execv(sys.executable, [sys.executable, *sys.argv])


def serve_http(port):
    """Streamable HTTP, answering each POST to /mcp with plain JSON."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path.rstrip("/") != "/mcp":
                return self.send_error(404)
            reply = reply_to(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
            if reply is None:
                self.send_response(202)
                self.end_headers()
                return
            body = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # no server-initiated stream
            self.send_error(405)

        def log_message(self, *_):
            pass

    print(f"duct-qa MCP on http://127.0.0.1:{port}/mcp", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    # stdio by default (spawned by .mcp.json); `--http [port]` to host it,
    # on 3101 by default, clear of the 3000-3010 another project's QA uses.
    if "--http" in sys.argv:
        rest = sys.argv[sys.argv.index("--http") + 1:]
        serve_http(int(rest[0]) if rest else int(os.environ.get("DUCT_QA_PORT", 3101)))
    else:
        serve_stdio()
