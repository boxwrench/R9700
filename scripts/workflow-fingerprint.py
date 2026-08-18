#!/usr/bin/env python3
"""
Workflow Semantic Fingerprint Generator and Validator for R9700.
Extracts semantic generation parameters while ignoring harmless UI/layout metadata.
"""

import sys
import os
import json
import hashlib

def normalize_workflow(data):
    """
    Normalizes a ComfyUI prompt/workflow dictionary into a stable canonical semantic structure.
    Handles both API prompt format (dict of nodes) and UI graph format (dict with 'nodes' array).
    """
    if "prompt" in data:
        data = data["prompt"]

    semantic = {
        "nodes": {},
        "models": [],
        "sampler_params": {},
        "dimensions": {}
    }

    if isinstance(data, dict) and not "nodes" in data:
        # API prompt format: {"1": {"class_type": ..., "inputs": ...}}
        for node_id in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x):
            node = data[node_id]
            ctype = node.get("class_type", "")
            inputs = node.get("inputs", {})
            
            clean_inputs = {}
            for k in sorted(inputs.keys()):
                v = inputs[k]
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    # Connection: [node_id, slot]
                    clean_inputs[k] = f"link({v[0]}:{v[1]})"
                else:
                    clean_inputs[k] = v
                    
                # Track key properties
                if k in ("unet_name", "clip_name", "vae_name", "model_name"):
                    if isinstance(v, str) and v not in semantic["models"]:
                        semantic["models"].append(v)
                if k in ("sampler_name", "scheduler", "steps", "denoise", "cfg_scale", "top_k"):
                    semantic["sampler_params"][k] = v
                if k in ("width", "height", "length", "seconds", "fps"):
                    semantic["dimensions"][k] = v

            semantic["nodes"][str(node_id)] = {
                "class_type": ctype,
                "inputs": clean_inputs
            }
    elif isinstance(data, dict) and "nodes" in data:
        # UI graph format
        for node in sorted(data["nodes"], key=lambda x: x.get("id", 0)):
            nid = str(node.get("id", 0))
            ctype = node.get("type", "")
            wvals = node.get("widgets_values", [])
            semantic["nodes"][nid] = {
                "class_type": ctype,
                "widgets_values": wvals
            }
            for val in wvals:
                if isinstance(val, str) and (val.endswith(".safetensors") or val.endswith(".gguf")):
                    if val not in semantic["models"]:
                        semantic["models"].append(val)

    semantic["models"].sort()
    return semantic

def compute_fingerprints(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Workflow not found: {file_path}")

    with open(file_path, "rb") as f:
        content = f.read()

    file_sha256 = hashlib.sha256(content).hexdigest()

    data = json.loads(content.decode("utf-8"))
    normalized = normalize_workflow(data)
    canonical_json = json.dumps(normalized, sort_keys=True, indent=2)
    semantic_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return {
        "file": file_path,
        "file_sha256": file_sha256,
        "semantic_sha256": semantic_sha256,
        "normalized": normalized
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: workflow-fingerprint.py <workflow.json> [expected_semantic_sha256]")
        sys.exit(1)

    wf_path = sys.argv[1]
    res = compute_fingerprints(wf_path)

    if len(sys.argv) >= 3:
        expected = sys.argv[2]
        if res["semantic_sha256"] == expected:
            print(f"[PASS] {wf_path} semantic fingerprint matches ({expected})")
            sys.exit(0)
        else:
            print(f"[FAIL] {wf_path} semantic fingerprint mismatch!")
            print(f"  Expected: {expected}")
            print(f"  Found:    {res['semantic_sha256']}")
            sys.exit(1)
    else:
        print(json.dumps({
            "file": res["file"],
            "file_sha256": res["file_sha256"],
            "semantic_sha256": res["semantic_sha256"],
            "models": res["normalized"]["models"],
            "sampler_params": res["normalized"]["sampler_params"],
            "dimensions": res["normalized"]["dimensions"]
        }, indent=2))

if __name__ == "__main__":
    main()
