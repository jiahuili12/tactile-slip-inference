"""Download version-pinned public CSVs; save hashes and upstream dataset card."""
import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone

from .common import ROOT, project_path, save_json

DATASET = "pollen-robotics/anyskin_slip_detection"
REVISION = "d9f5e315006482635a2effa34488715871d55c13"


def get_bytes(url):
    request = urllib.request.Request(url, headers={"User-Agent": "tactile-slip-inference/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Resolve a mutable revision to a commit before reading any data.
    info = json.loads(get_bytes(f"https://huggingface.co/api/datasets/{DATASET}/revision/{args.revision}"))
    revision = info["sha"]
    entries = json.loads(get_bytes(f"https://huggingface.co/api/datasets/{DATASET}/tree/{revision}/data"))
    files = []
    for entry in entries:
        if entry["type"] != "file" or not entry["path"].endswith(".csv"):
            continue
        name = entry["path"].split("/")[-1]
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{entry['path']}"
        payload = get_bytes(url)
        if len(payload) != entry["size"]:
            raise ValueError(f"Unexpected download size: {name}")
        (output / name).write_bytes(payload)
        files.append({"file": name, "source_url": url, "bytes": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest()})
        print(f"Downloaded {name}: {len(payload):,} bytes", flush=True)
    card_url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/README.md"
    (output / "UPSTREAM_README.md").write_bytes(get_bytes(card_url))
    save_json(ROOT / "data/source_manifest.json", {
        "dataset": DATASET, "revision": revision,
        "license_declared_by_dataset_card": info.get("cardData", {}).get("license", "apache-2.0"),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "files": files,
    })
    print(f"Saved provenance for {len(files)} CSV files.")


if __name__ == "__main__":
    main()
