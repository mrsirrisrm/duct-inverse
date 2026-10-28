"""Score Jev on mapping Revit family/type names to acoustic element types.

    python3 eval/jev_eval.py [out.json] [items.json]

Items default to element_types.json; labels always come from it.

Reads TYPESAFE_API_KEY from the environment or from the repo's .env.
Runs every item twice, with the label order reversed on the second pass
(Jev leans toward the first option), and reports accuracy, agreement between
passes, and accuracy by confidence band.
"""
import concurrent.futures
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).parent
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
INSTRUCTIONS = (
    "Which element type in a ductwork downduct noise calculation does this "
    "Revit element represent?"
)


def api_key():
    if key := os.environ.get("TYPESAFE_API_KEY"):
        return key
    env = HERE.parent.parent / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        name, _, value = line.partition("=")
        if name.strip() == "TYPESAFE_API_KEY":
            return value.strip().strip("\"'")
    sys.exit("TYPESAFE_API_KEY not set (environment or duct-inverse/.env)")


# Labels each Revit category can hold; the rest are never offered.
INLINE = {"elbow_mitred", "elbow_mitred_vanes", "elbow_radius", "branch_takeoff", "tee_or_wye", "transition", "end_cap",
          "coupling", "volume_damper", "fire_smoke_damper", "silencer", "plenum", "vav_terminal_unit", "fan", "other_unknown"}
ALLOWED = {
    "Duct Fittings": INLINE,
    "Duct Accessories": INLINE,
    "Air Terminals": {"diffuser_ceiling", "grille_or_register", "external_cap_or_louvre", "other_unknown"},
    "Mechanical Equipment": {"vav_terminal_unit", "fan", "air_handling_unit", "fan_coil_or_heat_pump", "not_airside",
                             "silencer", "plenum", "other_unknown"},
}


def ask(key, item, labels):
    allowed = ALLOWED.get(item["category"])
    labels = {k: v for k, v in labels.items() if allowed is None or k in allowed}
    body = json.dumps({
        "model": MODEL,
        "state": {"revit_category": item["category"], "family": item["family"], "type": item["type"]},
        "questions": {
            "element_type": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": labels},
        },
    }).encode()
    request = urllib.request.Request(
        URL, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    )
    for attempt in range(6):
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                reply = json.load(response)
            reply["latency_ms"] = (time.perf_counter() - start) * 1000
            return reply
        except urllib.error.HTTPError as e:
            if e.code not in (429, 529, 500, 502, 503) or attempt == 5:
                raise RuntimeError(f"{e.code}: {e.read().decode()[:300]}") from e
            time.sleep(2**attempt)


def run(key, items, labels):
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        return list(pool.map(lambda item: ask(key, item, labels), items))


def main():
    spec = json.loads((HERE / "element_types.json").read_text())
    labels = spec["labels"]
    items = json.loads(pathlib.Path(sys.argv[2]).read_text())["items"] if len(sys.argv) > 2 else spec["items"]
    key = api_key()
    forward = run(key, items, labels)
    reverse = run(key, items, dict(reversed(list(labels.items()))))

    rows = []
    for item, f, r in zip(items, forward, reverse):
        a, b = f["answers"]["element_type"], r["answers"]["element_type"]
        rows.append({
            **item,
            "choice": a["choice"],
            "confidence": a["confidence"],
            "choice_reversed": b["choice"],
            "confidence_reversed": b["confidence"],
            "top3": sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:3],
            "latency_ms": f["latency_ms"],
            "input_tokens": f["usage"]["input_tokens"],
        })
    model = forward[0]["model"]

    print(f"model {model}, {len(rows)} items, 2 passes")
    for name in sorted({r.get("set", "all") for r in rows}):
        subset = [r for r in rows if r.get("set", "all") == name]
        right = sum(r["choice"] in r["accept"] for r in subset)
        right_rev = sum(r["choice_reversed"] in r["accept"] for r in subset)
        print(f"  {name:9s} {right}/{len(subset)} correct (reversed order {right_rev}/{len(subset)})")
    agree = sum(r["choice"] == r["choice_reversed"] for r in rows)
    print(f"  passes agree on {agree}/{len(rows)}")

    print("accuracy by confidence (forward pass):")
    for lo, hi in ((0.9, 1.01), (0.5, 0.9), (0.0, 0.5)):
        band = [r for r in rows if lo <= r["confidence"] < hi]
        right = sum(r["choice"] in r["accept"] for r in band)
        print(f"  [{lo:.1f}, {min(hi, 1):.1f}]  {right}/{len(band)} correct")

    latency = sorted(r["latency_ms"] for r in rows)
    tokens = sum(r["input_tokens"] for r in rows) * 2
    print(f"latency p50 {latency[len(latency) // 2]:.0f} ms, p95 {latency[int(len(latency) * 0.95)]:.0f} ms; "
          f"{tokens} input tokens over both passes")

    print("misses and disagreements:")
    for r in rows:
        if r["choice"] not in r["accept"] or r["choice"] != r["choice_reversed"]:
            mark = "ok  " if r["choice"] in r["accept"] else "MISS"
            print(f"  {mark} {r['family']} | {r['type']}: {r['choice']} ({r['confidence']:.2f}), "
                  f"reversed {r['choice_reversed']} ({r['confidence_reversed']:.2f}); want {r['accept']}")

    if len(sys.argv) > 1:
        pathlib.Path(sys.argv[1]).write_text(json.dumps({"model": model, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
