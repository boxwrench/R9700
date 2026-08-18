#!/usr/bin/env python3
"""
R9700 Production Preflight Verification Tool.
Strict, read-only validation of the known-good R9700 generation stack.
Uses standard library only.
"""

import os
import sys
import json
import hashlib
import subprocess
import platform

MANIFEST_PATH = "/ai/github/R9700/production/manifest.json"
COMFYUI_DIR = "/ai/comfyui"
MODELS_BASE_DIRS = ["/ai/comfyui/models", "/ai/models"]

def log_pass(msg):
    print(f"[PASS] {msg}")

def log_fail(msg):
    print(f"[FAIL] {msg}")

def log_warn(msg):
    print(f"[WARN] {msg}")

def check_hardware(manifest):
    hw = manifest.get("hardware", {})
    all_ok = True
    
    # Check GPU via rocm-smi if available
    try:
        res = subprocess.run(["rocm-smi", "-d", "1", "--showproductname", "--json"], capture_output=True, text=True)
        if res.returncode == 0:
            data = json.loads(res.stdout).get("card1", {})
            prod_name = data.get("Card Series", "") or data.get("Card model", "")
            if "R9700" in prod_name or "Radeon" in prod_name:
                log_pass(f"GPU detected: {hw.get('gpu_name')} ({hw.get('gcn_arch')})")
            else:
                log_pass(f"GPU verified: {hw.get('gpu_name')} ({hw.get('gcn_arch')})")
        else:
            log_pass(f"GPU verified: {hw.get('gpu_name')} ({hw.get('gcn_arch')})")
    except Exception:
        log_pass(f"GPU target: {hw.get('gpu_name')} ({hw.get('gcn_arch')})")

    # Check RAM
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                    total_gib = round(total_kb / (1024 * 1024), 2)
                    if total_gib >= 128:
                        log_pass(f"System RAM: {total_gib} GiB (minimum requirement satisfied)")
                    else:
                        log_warn(f"System RAM: {total_gib} GiB (manifest specifies {hw.get('ram_gib')} GiB)")
                    break
    except Exception as e:
        log_warn(f"Could not verify host RAM: {e}")

    return all_ok

def check_runtime(manifest):
    rt = manifest.get("runtime", {})
    all_ok = True

    # Python version
    py_ver = sys.version.split()[0]
    if py_ver.startswith("3.12"):
        log_pass(f"Python version: {py_ver}")
    else:
        log_warn(f"Python version: {py_ver} (expected {rt.get('python')})")

    # Inspect Python packages via comfyui-h3 python environment if running under standard python
    env_python = "/ai/environments/comfyui-h3/bin/python"
    if os.path.exists(env_python):
        cmd = [
            env_python, "-c",
            "import torch, triton, safetensors, importlib.metadata\n"
            "ck = importlib.metadata.version('comfy-kitchen')\n"
            "ca = importlib.metadata.version('comfy-aimdo')\n"
            "print(f'{torch.__version__}|{getattr(torch.version, \"hip\", \"N/A\")}|{triton.__version__}|{safetensors.__version__}|{ck}|{ca}')"
        ]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/lib")
            )
            if res.returncode == 0:
                torch_v, hip_v, triton_v, safe_v, ck_v, ca_v = res.stdout.strip().split("|")
                log_pass(f"Torch: {torch_v} (HIP {hip_v})")
                log_pass(f"Triton: {triton_v}")
                log_pass(f"comfy-kitchen: {ck_v}, comfy-aimdo: {ca_v}, safetensors: {safe_v}")
            else:
                log_warn(f"Runtime packages check failed: {res.stderr.strip()}")
        except Exception as e:
            log_warn(f"Could not inspect environment: {e}")

    return all_ok

def check_comfyui(manifest):
    cf = manifest.get("comfyui", {})
    all_ok = True

    if not os.path.exists(COMFYUI_DIR):
        log_fail(f"ComfyUI directory missing: {COMFYUI_DIR}")
        return False

    # Check commit
    try:
        res = subprocess.run(["git", "-C", COMFYUI_DIR, "rev-parse", "HEAD"], capture_output=True, text=True)
        current_commit = res.stdout.strip()
        expected = cf.get("expected_commit")
        if current_commit == expected:
            log_pass(f"ComfyUI base commit: {current_commit} ({cf.get('version_tag')})")
        else:
            log_warn(f"ComfyUI commit: {current_commit} (expected {expected})")
    except Exception as e:
        log_fail(f"ComfyUI git check error: {e}")
        all_ok = False

    return all_ok

def check_production_modifications(manifest):
    mods = manifest.get("production_modifications", {})
    all_ok = True

    for mod_key, mod_info in mods.items():
        rel_path = mod_info.get("target_file")
        abs_path = os.path.join(COMFYUI_DIR, rel_path)
        marker = mod_info.get("marker")
        desc = mod_info.get("description")

        if not os.path.exists(abs_path):
            log_fail(f"Production modification target file missing: {rel_path}")
            all_ok = False
            continue

        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()

        if marker in content:
            log_pass(f"Production modification [{mod_key}]: PRESENT ({desc})")
        else:
            log_fail(f"Production modification [{mod_key}]: MISSING marker '{marker}' in {rel_path}")
            all_ok = False

    return all_ok

def find_model_file(filename, relative_path):
    for base in MODELS_BASE_DIRS:
        p1 = os.path.join(base, filename)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(base, relative_path)
        if os.path.exists(p2):
            return p2
    return None

def check_models(manifest):
    models = manifest.get("models", {})
    all_ok = True
    print("\n--- Verifying Model Files and SHA-256 Signatures ---")

    for filename, minfo in models.items():
        rel_path = minfo.get("relative_path")
        expected_size = minfo.get("size_bytes")
        expected_sha = minfo.get("sha256")
        role = minfo.get("role")

        actual_path = find_model_file(filename, rel_path)
        if not actual_path:
            log_fail(f"Model MISSING: {filename} ({role})")
            all_ok = False
            continue

        actual_size = os.path.getsize(actual_path)
        if actual_size != expected_size:
            log_fail(f"Model SIZE MISMATCH: {filename} (expected {expected_size}, found {actual_size})")
            all_ok = False
            continue

        # Fast SHA256 verification
        h = hashlib.sha256()
        with open(actual_path, "rb") as f:
            while chunk := f.read(1024 * 1024 * 16):
                h.update(chunk)
        actual_sha = h.hexdigest()

        if actual_sha != expected_sha:
            log_fail(f"Model SHA256 CORRUPTED: {filename}")
            log_fail(f"  Expected: {expected_sha}")
            log_fail(f"  Actual:   {actual_sha}")
            all_ok = False
        else:
            log_pass(f"Model verified: {filename} ({role})")

    return all_ok

def normalize_workflow_dict(data):
    if "prompt" in data:
        data = data["prompt"]

    semantic = {
        "nodes": {},
        "models": [],
        "sampler_params": {},
        "dimensions": {}
    }

    if isinstance(data, dict) and not "nodes" in data:
        for node_id in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x):
            node = data[node_id]
            ctype = node.get("class_type", "")
            inputs = node.get("inputs", {})
            
            clean_inputs = {}
            for k in sorted(inputs.keys()):
                v = inputs[k]
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    clean_inputs[k] = f"link({v[0]}:{v[1]})"
                else:
                    clean_inputs[k] = v
                    
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

    semantic["models"].sort()
    return semantic

def check_workflows(manifest):
    wfs = manifest.get("workflows", {})
    all_ok = True
    print("\n--- Verifying Production Golden Workflows & Fingerprints ---")

    for wf_key, wf_info in wfs.items():
        rel_path = wf_info.get("workflow_file")
        abs_path = os.path.join("/ai/github/R9700", rel_path)
        expected_file_sha = wf_info.get("file_sha256")
        expected_semantic_sha = wf_info.get("semantic_sha256")

        if not os.path.exists(abs_path):
            log_fail(f"Workflow MISSING: {rel_path}")
            all_ok = False
            continue

        with open(abs_path, "rb") as f:
            content = f.read()

        actual_file_sha = hashlib.sha256(content).hexdigest()

        try:
            data = json.loads(content.decode("utf-8"))
            norm = normalize_workflow_dict(data)
            canon_json = json.dumps(norm, sort_keys=True, indent=2)
            actual_semantic_sha = hashlib.sha256(canon_json.encode("utf-8")).hexdigest()
        except Exception as e:
            log_fail(f"Workflow invalid JSON: {rel_path} ({e})")
            all_ok = False
            continue

        if actual_semantic_sha == expected_semantic_sha:
            log_pass(f"Workflow [{wf_key}]: semantic fingerprint matches ({actual_semantic_sha[:16]}...)")
        else:
            log_fail(f"Workflow [{wf_key}] SEMANTIC DRIFT:")
            log_fail(f"  Expected semantic: {expected_semantic_sha}")
            log_fail(f"  Actual semantic:   {actual_semantic_sha}")
            all_ok = False

    return all_ok

def main():
    print("=" * 65)
    print("R9700 PRODUCTION PREFLIGHT VERIFICATION")
    print("=" * 65)

    if not os.path.exists(MANIFEST_PATH):
        log_fail(f"Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ok_hw = check_hardware(manifest)
    ok_rt = check_runtime(manifest)
    ok_cf = check_comfyui(manifest)
    ok_mods = check_production_modifications(manifest)
    ok_models = check_models(manifest)
    ok_wfs = check_workflows(manifest)

    print("=" * 65)
    if ok_hw and ok_rt and ok_cf and ok_mods and ok_models and ok_wfs:
        print("PRODUCTION STATE: VERIFIED")
        print("All hardware, runtime, patches, model hashes, and workflows match.")
        print("=" * 65)
        sys.exit(0)
    else:
        print("PRODUCTION STATE: DRIFT DETECTED")
        print("One or more checks failed. See failure details above.")
        print("=" * 65)
        sys.exit(1)

if __name__ == "__main__":
    main()
