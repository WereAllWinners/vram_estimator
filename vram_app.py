import math
import re
import urllib.parse
import streamlit as st
from dataclasses import dataclass
import pandas as pd
from huggingface_hub import list_models
from transformers import AutoConfig
import ollama
import requests
from bs4 import BeautifulSoup
from torchinfo import summary
import torch
import altair as alt
import plotly.express as px
import GPUtil
import subprocess
import platform
import json
import os
import glob as _glob

st.set_page_config(page_title="VRAM Estimator", page_icon="🧠", layout="wide")

# Hardware detection session state
if "hw_detected_gpus" not in st.session_state:
    st.session_state["hw_detected_gpus"] = []

# Model + math helpers
BYTES_PER = {"FP4": 0.5, "FP8": 1.0, "FP16": 2.0, "FP32": 4.0}
UNIT_DIVISORS = {"GB (decimal, 1e9)": 1e9, "GiB (binary, 1024^3)": 1024**3}

@dataclass
class ModelShape:
    params_bil: float
    layers: int
    hidden: int

def to_units(x_bytes: float, divisor: float) -> float:
    return x_bytes / divisor

def estimate_memory(
    task_type: str,
    shape: ModelShape,
    weight_precision: str = "FP16",
    kv_precision: str = "FP16",
    act_precision: str = "FP16",
    seq_len: int = 4096,
    batch_tokens: int = 1,
    lora_rank: int = 32,
    unit_divisor: float = 1e9,
    num_gpus: int = 1,
    parallelism_type: str = "None",
    benchmark: bool = False,
) -> dict:
    params = shape.params_bil * 1e9
    w_b = BYTES_PER[weight_precision]
    kv_b = BYTES_PER[kv_precision]
    act_b = BYTES_PER[act_precision]

    weights = params * w_b
    tokens_in_flight = max(1, seq_len * batch_tokens)
    kv_per_token_bytes = 2 * shape.layers * shape.hidden * kv_b
    kv_cache = kv_per_token_bytes * tokens_in_flight

    optimizer = grads = master_weights = lora_overhead = 0.0
    if task_type in {"Full-Fine-Tuning", "Training"}:
        optimizer = params * 8.0
        grads = params * 2.0
        master_weights = params * 2.0
    if task_type == "LoRA":
        lora_params = params * 0.01 * (lora_rank / 32.0)
        lora_overhead = lora_params * w_b

    if benchmark:
        class DummyModel(torch.nn.Module):
            def __init__(self): super().__init__(); self.fc = torch.nn.Linear(shape.hidden, shape.hidden)
        model = DummyModel()
        input_data = torch.randn(batch_tokens, seq_len, shape.hidden)
        info = summary(model, input_data=input_data, verbose=0)
        activations = info.total_mult_adds * act_b / 1e9
    else:
        activations_scale = max(batch_tokens / max(1, seq_len), 0.25)
        activations = (params * act_b) * activations_scale

    total_bytes = weights + kv_cache + lora_overhead + activations + optimizer + grads + master_weights

    if parallelism_type == "Tensor Parallel":
        total_bytes /= num_gpus
        kv_cache /= num_gpus
    elif parallelism_type == "Pipeline Parallel":
        total_bytes = (total_bytes / num_gpus) + kv_cache
    elif parallelism_type == "Data Parallel":
        total_bytes *= num_gpus

    per_gpu = total_bytes / num_gpus if num_gpus > 1 else total_bytes

    def conv(x): return round(to_units(x, unit_divisor), 2)
    return {
        "weights": conv(weights), "kv_cache": conv(kv_cache), "lora_overhead": conv(lora_overhead),
        "activations": conv(activations), "optimizer": conv(optimizer), "grads": conv(grads),
        "master_weights": conv(master_weights), "total": conv(total_bytes), "per_gpu": conv(per_gpu),
    }

# ─── GPU VRAM Database ────────────────────────────────────────────────────────
# VRAM in GB. APU entries = max GPU-accessible portion of unified system RAM.
GPU_VRAM_DB = {
    # === NVIDIA Blackwell ===
    "GB10":            128,   # DGX Spark – 128 GB LPDDR5X unified (CPU+GPU)
    "GB200":           384,   # Grace Blackwell NVL72
    "B200":            192,
    "B100":            192,
    "RTX 5090":         32,
    "RTX 5080":         16,
    "RTX 5070 Ti":      16,
    "RTX 5070":         12,
    "RTX 5060 Ti":      16,
    "RTX 5060":          8,
    # === NVIDIA Ada Lovelace (RTX 40xx) ===
    "RTX 4090":         24,
    "RTX 4080 Super":   16,
    "RTX 4080":         16,
    "RTX 4070 Ti Super":16,
    "RTX 4070 Ti":      12,
    "RTX 4070 Super":   12,
    "RTX 4070":         12,
    "RTX 4060 Ti 16GB": 16,
    "RTX 4060 Ti":       8,
    "RTX 4060":          8,
    "RTX 4050":          6,
    # === NVIDIA Ampere (RTX 30xx) ===
    "RTX 3090 Ti":      24,
    "RTX 3090":         24,
    "RTX 3080 Ti":      12,
    "RTX 3080 12GB":    12,
    "RTX 3080":         10,
    "RTX 3070 Ti":       8,
    "RTX 3070":          8,
    "RTX 3060 Ti":       8,
    "RTX 3060":         12,
    "RTX 3050 8GB":      8,
    "RTX 3050":          4,
    # === NVIDIA Turing (RTX 20xx / GTX 16xx) ===
    "RTX 2080 Ti":      11,
    "RTX 2080 Super":    8,
    "RTX 2080":          8,
    "RTX 2070 Super":    8,
    "RTX 2070":          8,
    "RTX 2060 Super":    8,
    "RTX 2060":          6,
    "GTX 1660 Super":    6,
    "GTX 1660 Ti":       6,
    "GTX 1660":          6,
    "GTX 1650 Super":    4,
    "GTX 1650":          4,
    # === NVIDIA Data Center / HPC ===
    "H200":            141,
    "H100 80GB":        80,
    "H100":             80,
    "A100 80GB":        80,
    "A100":             40,
    "A40":              48,
    "A30":              24,
    "A10G":             24,
    "A10":              24,
    "L40S":             48,
    "L40":              48,
    "L4":               24,
    "V100 32GB":        32,
    "V100":             16,
    "T4":               16,
    "P100 16GB":        16,
    "P100":             12,
    # === NVIDIA Quadro / RTX Professional ===
    "RTX 6000 Ada":     48,
    "RTX 5000 Ada":     32,
    "RTX 4500 Ada":     24,
    "RTX 4000 Ada":     20,
    "RTX A6000":        48,
    "RTX A5500":        24,
    "RTX A5000":        24,
    "RTX A4500":        20,
    "RTX A4000":        16,
    "RTX A2000 12GB":   12,
    "RTX A2000":         6,
    # === AMD RDNA 4 ===
    "RX 9070 XT":       16,
    "RX 9070":          16,
    "RX 9060 XT":       16,
    "RX 9060":           8,
    # === AMD RDNA 3 ===
    "RX 7900 XTX":      24,
    "RX 7900 XT":       20,
    "RX 7900 GRE":      16,
    "RX 7800 XT":       16,
    "RX 7700 XT":       12,
    "RX 7600 XT":       16,
    "RX 7600":           8,
    "RX 7500 XT":        8,
    # === AMD RDNA 2 ===
    "RX 6950 XT":       16,
    "RX 6900 XT":       16,
    "RX 6800 XT":       16,
    "RX 6800":          16,
    "RX 6750 XT":       12,
    "RX 6700 XT":       12,
    "RX 6700":          10,
    "RX 6650 XT":        8,
    "RX 6600 XT":        8,
    "RX 6600":           8,
    "RX 6500 XT":        4,
    "RX 6400":           4,
    # === AMD APU / Integrated — Strix Halo (RDNA 3.5) ===
    # Strix Halo uses LPDDR5X unified memory shared with CPU.
    # GPU can access up to ~96 GB on 128 GB configs (OS/CPU reserve ~32 GB).
    "Radeon 8060S":     96,   # Strix Halo iGPU (Ryzen AI Max), 40 CU RDNA 3.5
    "Radeon 8050S":     64,   # Strix Halo lower tier
    "Radeon 890M":      96,   # Alt branding for Strix Halo 40 CU iGPU
    "Radeon 880M":      24,   # Strix Point / Hawk Point
    "Radeon 870M":      16,
    "Radeon 860M":      12,
    "Radeon 850M":       8,
    "Radeon 780M":      16,   # Phoenix
    "Radeon 760M":       8,
    "Radeon 740M":       4,
    "Vega 11":          16,
    "Vega 8":           16,
    "Vega 7":           16,
    # === AMD Instinct / Data Center ===
    "MI350X":          288,
    "MI325X":          288,
    "MI300X":          192,
    "MI300A":          128,
    "MI250X":          128,
    "MI250":           128,
    "MI210":            64,
    "MI100":            32,
    # === Intel Arc ===
    "Arc B580":         12,
    "Arc B570":         10,
    "Arc A770 16GB":    16,
    "Arc A770":          8,
    "Arc A750":          8,
    "Arc A580":          8,
    "Arc A380":          6,
    "Arc A310":          4,
}

# Keywords that identify an APU / integrated / unified-memory GPU
_APU_KEYWORDS = {
    "890M", "880M", "870M", "860M", "850M",
    "780M", "760M", "740M", "8060S", "8050S",
    "VEGA 8", "VEGA 7", "VEGA 11",
    "IRIS XE", "IRIS PLUS", "UHD GRAPHICS",
}


def _fuzzy_vram_lookup(gpu_name: str):
    """Return VRAM (GB) from GPU_VRAM_DB via longest substring match, or None."""
    name_upper = gpu_name.upper()
    best_vram, best_len = None, 0
    for key, vram in GPU_VRAM_DB.items():
        if key.upper() in name_upper and len(key) > best_len:
            best_vram, best_len = vram, len(key)
    return best_vram


def detect_gpus() -> list:
    """
    Multi-method GPU detection. Supports:
      - NVIDIA (nvidia-smi, GPUtil)        — discrete + GB10 Superchip (DGX Spark)
      - AMD discrete (rocm-smi)            — RX 6000/7000/9000 series
      - AMD APU / Strix Halo (WMI/sysfs)  — Radeon 8060S, 890M, etc.
      - Intel Arc (WMI / sysfs)
    Returns list of dicts: {name, vram_gb, gpu_type, method, notes}
    """
    results, seen = [], set()

    def _add(name, vram_gb, gpu_type, method, notes=""):
        key = name.strip().lower()
        if key not in seen and vram_gb and vram_gb > 0:
            seen.add(key)
            results.append({
                "name": name.strip(),
                "vram_gb": round(float(vram_gb), 1),
                "gpu_type": gpu_type,
                "method": method,
                "notes": notes,
            })

    # ── 1. nvidia-smi (most reliable for NVIDIA + GB10 Superchip) ────────────
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                    name = parts[0]
                    vram_gb = int(parts[1]) / 1024          # MiB → GB
                    note = ("Unified memory – CPU+GPU share 128 GB LPDDR5X"
                            if "GB10" in name.upper() else "")
                    _add(name, vram_gb, "NVIDIA", "nvidia-smi", note)
    except Exception:
        pass

    # ── 2. GPUtil fallback (NVIDIA only, no nvidia-smi needed) ───────────────
    if not any(g["gpu_type"] == "NVIDIA" for g in results):
        try:
            for g in GPUtil.getGPUs():
                _add(g.name, g.memoryTotal / 1024, "NVIDIA", "GPUtil")
        except Exception:
            pass

    # ── 3. rocm-smi (AMD discrete GPUs with ROCm driver) ─────────────────────
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            for card_id, info in data.items():
                if not isinstance(info, dict):
                    continue
                name = (info.get("Card series")
                        or info.get("Card model")
                        or f"AMD GPU ({card_id})")
                vram_b = int(info.get("VRAM Total Memory (B)", 0))
                if vram_b:
                    _add(name, vram_b / 1e9, "AMD", "rocm-smi")
    except Exception:
        pass

    # ── 4. Windows WMI — covers AMD APUs, iGPUs, Intel Arc ───────────────────
    if platform.system() == "Windows":
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-WmiObject Win32_VideoController | "
                 "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"],
                capture_output=True, text=True, timeout=20,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                raw = json.loads(proc.stdout)
                if isinstance(raw, dict):
                    raw = [raw]
                for item in raw:
                    name = (item.get("Name") or "").strip()
                    if not name or name.lower() in seen:
                        continue
                    nu = name.upper()
                    gpu_type = (
                        "NVIDIA" if any(x in nu for x in ["NVIDIA", "GEFORCE", "QUADRO", "TESLA"]) else
                        "AMD"    if any(x in nu for x in ["AMD", "RADEON", "VEGA"]) else
                        "Intel"  if any(x in nu for x in ["INTEL", "ARC", "IRIS", "UHD"]) else
                        "Unknown"
                    )
                    is_apu = any(kw in nu for kw in _APU_KEYWORDS)
                    db_vram = _fuzzy_vram_lookup(name)
                    wmi_ram = item.get("AdapterRAM") or 0

                    if db_vram:
                        vram_gb = db_vram
                        note = (
                            "APU / unified memory — GPU shares system LPDDR5X RAM; "
                            "max VRAM depends on total RAM and BIOS allocation"
                            if is_apu else "VRAM from database"
                        )
                    elif wmi_ram and wmi_ram > 256 * 1024 * 1024:
                        vram_gb = wmi_ram / 1e9
                        note = "VRAM reported by WMI driver"
                    else:
                        continue   # can't determine VRAM — skip

                    _add(name, vram_gb, gpu_type, "WMI", note)
        except Exception:
            pass

    # ── 5. Linux sysfs (AMD discrete + APU) ──────────────────────────────────
    if platform.system() == "Linux":
        for mem_path in _glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
            dev = os.path.dirname(mem_path)
            try:
                vendor_path = os.path.join(dev, "vendor")
                with open(vendor_path) as f:
                    if f.read().strip() != "0x1002":
                        continue
                with open(mem_path) as f:
                    vram_b = int(f.read().strip())
                name = "AMD GPU"
                uevent = os.path.join(dev, "uevent")
                if os.path.exists(uevent):
                    with open(uevent) as f:
                        for line in f:
                            if line.startswith("PCI_ID="):
                                name = f"AMD GPU [{line.split('=')[1].strip()}]"
                db_vram = _fuzzy_vram_lookup(name)
                _add(name, db_vram or vram_b / 1e9, "AMD", "sysfs")
            except Exception:
                continue

    return results


# UI
st.title("🧠 VRAM Requirement Estimator")
st.caption("Weights • KV cache • Optimizer/Grads • Activations | quick, practical sizing (rules of thumb)")

with st.sidebar:
    st.header("Inputs")
    unit_label = st.selectbox("Display units", list(UNIT_DIVISORS.keys()), index=1)
    unit_divisor = UNIT_DIVISORS[unit_label]

    st.subheader("Model Selection")
    tab1, tab2, tab3 = st.tabs(["Hugging Face Search", "Ollama Models", "Custom"])

    params_bil, layers, hidden = 8.0, 32, 4096  # Defaults
    with tab1:
        search_query = st.text_input("Search HF models (e.g., 'llama 3 latest')")
        if search_query:
            models = list_models(search=search_query, filter="text-generation", sort="downloads", direction=-1, limit=20)
            model_options = [m.id for m in models]
            selected_model = st.selectbox("Select HF model", model_options)
            if selected_model:
                try:
                    config = AutoConfig.from_pretrained(selected_model)
                    params_bil = config.num_params / 1e9 if hasattr(config, 'num_params') else params_bil
                    layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else layers
                    hidden = config.hidden_size if hasattr(config, 'hidden_size') else hidden
                except:
                    st.error("Config load failed; use defaults or Custom.")

    with tab2:
        # ── Local installed models ──────────────────────────────────────────
        st.markdown("**Local Models**")
        if st.button("Fetch Local Ollama Models"):
            try:
                response = ollama.list()
                st.session_state['ollama_models'] = [m.model for m in response.models]
            except Exception as e:
                st.error(f"Failed to connect to Ollama: {e}")

        if st.session_state.get('ollama_models'):
            selected_ollama = st.selectbox("Select local model", st.session_state['ollama_models'])
            if selected_ollama:
                try:
                    details = ollama.show(selected_ollama)
                    model_details = getattr(details, 'details', None)
                    param_size_str = getattr(model_details, 'parameter_size', '8B') or '8B'
                    params_bil = float(param_size_str.upper().replace('B', '').strip() or 8.0)
                    model_info = getattr(details, 'model_info', {}) or {}
                    layers = int(model_info.get('llm.block_count', 32))
                    hidden = int(model_info.get('llm.embedding_length', 4096))
                    st.session_state['ollama_selected_params'] = (params_bil, layers, hidden)
                except Exception as e:
                    st.error(f"Failed to load model details: {e}")

        st.divider()

        # ── Browse Ollama Library ───────────────────────────────────────────
        st.markdown("**Browse Ollama Library**")

        def _fetch_ollama_search(query, sort):
            sort_param = "newest" if sort == "Newest" else "popular"
            url = f"https://ollama.com/search?q={urllib.parse.quote(query)}&sort={sort_param}"
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for a in soup.find_all("a", href=lambda h: h and h.startswith("/library/")):
                name_el = a.find("span", attrs={"x-test-search-response-title": True})
                if not name_el:
                    continue
                sizes = [s.get_text(strip=True) for s in a.find_all("span", attrs={"x-test-size": True})]
                caps  = [s.get_text(strip=True) for s in a.find_all("span", attrs={"x-test-capability": True})]
                pulls_el = a.find("span", attrs={"x-test-pull-count": True})
                results.append({
                    "name":  name_el.get_text(strip=True),
                    "sizes": sizes,
                    "tags":  caps,
                    "pulls": pulls_el.get_text(strip=True) if pulls_el else "",
                })
            return results

        col_q, col_s = st.columns([3, 1])
        with col_q:
            lib_query = st.text_input("Search", placeholder="e.g. llama, qwen, mistral", key="lib_query")
        with col_s:
            lib_sort = st.selectbox("Sort", ["Popular", "Newest"], key="lib_sort")

        if st.button("Search Library"):
            if lib_query.strip():
                try:
                    with st.spinner("Searching…"):
                        st.session_state["lib_results"] = _fetch_ollama_search(lib_query.strip(), lib_sort)
                        st.session_state["lib_search_key"] = lib_query.strip()
                except Exception as e:
                    st.error(f"Search failed: {e}")
            else:
                st.warning("Enter a search term first.")

        lib_results = st.session_state.get("lib_results", [])
        if lib_results:
            # Filter controls
            all_sizes = sorted(set(s for m in lib_results for s in m["sizes"]))
            all_tags  = sorted(set(t for m in lib_results for t in m["tags"]))
            fc1, fc2 = st.columns(2)
            with fc1:
                size_filter = st.multiselect("Filter by size", all_sizes, key="lib_size_filter")
            with fc2:
                tag_filter = st.multiselect("Filter by capability", all_tags, key="lib_tag_filter")

            filtered = lib_results
            if size_filter:
                filtered = [m for m in filtered if any(s in m["sizes"] for s in size_filter)]
            if tag_filter:
                filtered = [m for m in filtered if any(t in m["tags"] for t in tag_filter)]

            if filtered:
                df_lib = pd.DataFrame([{
                    "Model": m["name"],
                    "Sizes": ", ".join(m["sizes"]) or "—",
                    "Capabilities": ", ".join(m["tags"]) or "—",
                    "Pulls": m["pulls"],
                } for m in filtered])
                st.dataframe(df_lib, use_container_width=True, hide_index=True)

                sel_name = st.selectbox("Select model", [m["name"] for m in filtered], key="lib_sel_name")
                sel_data = next((m for m in filtered if m["name"] == sel_name), None)

                if sel_data:
                    if sel_data["sizes"]:
                        sel_size = st.selectbox("Select size", sel_data["sizes"], key="lib_sel_size")
                        # Parse param count from size string (e.g. "70b" → 70.0, "0.5b" → 0.5)
                        m = re.match(r"(\d+\.?\d*)([bBmMkK])", sel_size.split("x")[-1])
                        if m:
                            val, unit = float(m.group(1)), m.group(2).lower()
                            params_bil = val if unit == "b" else (val / 1e3 if unit == "m" else val / 1e6)
                            st.session_state["ollama_selected_params"] = (params_bil, layers, hidden)
                    else:
                        st.info("No size tags — set parameters in the Custom tab.")
            else:
                st.info("No models match the current filters.")

    with tab3:
        _sel = st.session_state.get("ollama_selected_params", (8.0, 32, 4096))
        params_bil = st.number_input("Parameters (B)", min_value=0.1, value=float(_sel[0]))
        layers = st.number_input("Layers", min_value=1, value=int(_sel[1]))
        hidden = st.number_input("Hidden size", min_value=256, value=int(_sel[2]))

    # Use session-state values from tab1/tab2 if set, else fall back to tab3 inputs
    if "ollama_selected_params" in st.session_state:
        params_bil, layers, hidden = st.session_state["ollama_selected_params"]

    shape = ModelShape(params_bil, layers, hidden)

    task = st.selectbox("Task type", ["Inference", "LoRA", "Full-Fine-Tuning", "Training"])
    colp = st.columns(3)
    with colp[0]: wprec = st.selectbox("Weights precision", list(BYTES_PER.keys()), index=2)
    with colp[1]: kvprec = st.selectbox("KV precision", list(BYTES_PER.keys()), index=2)
    with colp[2]: actprec = st.selectbox("Activations precision", list(BYTES_PER.keys()), index=2)

    seq_len = st.slider("Sequence length", 512, 65536, 8192, 512)
    batch_tokens = st.slider("Batch tokens", 1, 2048, 2, 1)
    lora_rank = 32 if task != "LoRA" else st.slider("LoRA rank", 4, 256, 32, 4)

    st.subheader("Parallelism & Hardware")
    num_gpus = st.slider("Number of GPUs", 1, 8, 1)

    # ── Detected GPU picker (populated after "Detect Hardware") ───────────────
    _detected = st.session_state.get("hw_detected_gpus", [])
    if _detected:
        _labels = [
            f"{g['name']}  ({g['vram_gb']} GB  ·  {g['gpu_type']}  ·  {g['method']})"
            for g in _detected
        ]
        _sel_idx = st.selectbox(
            "Detected GPU", range(len(_labels)),
            format_func=lambda i: _labels[i], key="hw_gpu_sel",
        )
        _sel = _detected[_sel_idx]
        if _sel.get("notes"):
            st.caption(f"ℹ️ {_sel['notes']}")
        _vram_default = float(_sel["vram_gb"])
    else:
        _vram_default = 48.0

    per_gpu_vram = st.number_input(
        f"VRAM per GPU ({unit_label.split()[0]})",
        min_value=0.0, value=_vram_default, step=1.0,
    )
    available_vram = per_gpu_vram * num_gpus

    parallelism_type = st.selectbox("Strategy", ["None", "Tensor Parallel", "Pipeline Parallel", "Data Parallel"])

    benchmark = st.checkbox("Run Benchmark for Activations (Slow)")

    # Show detection results banner (set before st.rerun() in button handler)
    if st.session_state.get("hw_just_detected"):
        for g in _detected:
            note_str = f"  — {g['notes']}" if g.get("notes") else ""
            st.success(f"✅ {g['gpu_type']}: **{g['name']}** · {g['vram_gb']} GB{note_str}")
        del st.session_state["hw_just_detected"]

    if st.button("🔍 Detect Hardware"):
        with st.spinner("Scanning for GPUs…"):
            _found = detect_gpus()
        if _found:
            st.session_state["hw_detected_gpus"] = _found
            st.session_state["hw_just_detected"] = True
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        else:
            st.warning(
                "No GPU detected. Possible reasons: no discrete GPU, "
                "missing drivers (nvidia-smi / ROCm), or running in a VM. "
                "Enter VRAM manually above."
            )

# Compute
res = estimate_memory(task, shape, wprec, kvprec, actprec, seq_len, batch_tokens, lora_rank, unit_divisor, num_gpus, parallelism_type, benchmark)

fit = res["total"] <= available_vram
status = "✅ Fits" if fit else "❌ Exceeds"
status_color = "green" if fit else "red"
if res["total"] > available_vram:
    st.warning("Exceeds VRAM! Try quantization or checkpointing.")
if wprec == "FP4":
    st.warning("FP4 may not support all hardware.")

# Output
left, right = st.columns([2, 1])
with left:
    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Total Required ({unit_label.split()[0]})", res["total"])
    m2.metric(f"Per GPU Required", res["per_gpu"])
    m3.metric(f"Total Available", available_vram)
    m4.metric("Headroom", available_vram - res["total"])
    st.markdown(f"**Status:** <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)

    chart_data = pd.DataFrame({
        "Component": ["Weights", "KV", "Optimizer", "Grads", "Master", "Activations", "LoRA"],
        "VRAM": [res["weights"], res["kv_cache"], res["optimizer"], res["grads"], res["master_weights"], res["activations"], res["lora_overhead"]]
    })
    chart = alt.Chart(chart_data).mark_bar().encode(x="Component", y="VRAM", color="Component").interactive()
    st.altair_chart(chart, use_container_width=True)

    seq_range = range(512, 16384, 1024)
    vram_by_seq = [estimate_memory(task, shape, wprec, kvprec, actprec, s, batch_tokens, lora_rank, unit_divisor, num_gpus, parallelism_type, benchmark)["total"] for s in seq_range]
    fig = px.line(x=seq_range, y=vram_by_seq, labels={"x": "Seq Len", "y": "VRAM"})
    st.plotly_chart(fig)

with right:
    st.subheader(f"Breakdown ({unit_label.split()[0]})")
    st.dataframe(res, use_container_width=True)

    if st.button("Export CSV"):
        df = pd.DataFrame([res])
        csv = df.to_csv(index=False)
        st.download_button("Download", csv, "vram_report.csv", "text/csv")

st.divider()
st.subheader("🔍 Model Discovery")
st.caption("Find Ollama and HuggingFace models that fit your available VRAM")

with st.expander("Configure & Run Discovery", expanded=False):
    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        disc_vram = st.number_input("Your VRAM (GB)", min_value=1.0, value=float(per_gpu_vram), step=1.0, key="disc_vram")
    with dc2:
        disc_prec = st.selectbox("Precision", list(BYTES_PER.keys()), index=2, key="disc_prec")
    with dc3:
        disc_pages = st.slider("Ollama pages to scan", 1, 20, 5, key="disc_pages",
                               help="Each page = 20 models from ollama.com/search sorted by popularity")
    with dc4:
        disc_overhead = st.slider("Overhead factor", 1.0, 2.0, 1.25, 0.05, key="disc_overhead",
                                  help="Multiplier for KV cache + activation overhead on top of raw weights")

    disc_source = st.multiselect(
        "Sources to scan",
        ["Ollama Library", "HuggingFace"],
        default=["Ollama Library"],
        key="disc_source",
    )

    if st.button("Find Compatible Models", key="disc_btn"):
        bytes_per = BYTES_PER[disc_prec]
        max_params_bil = (disc_vram * 1e9) / (bytes_per * disc_overhead) / 1e9
        compatible = []

        if "Ollama Library" in disc_source:
            prog = st.progress(0, text="Scanning Ollama library…")
            for i in range(disc_pages):
                prog.progress((i + 1) / disc_pages, text=f"Scanning Ollama page {i+1}/{disc_pages}…")
                try:
                    resp = requests.get(
                        f"https://ollama.com/search?p={i+1}",
                        timeout=10, headers={"User-Agent": "Mozilla/5.0"},
                    )
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for a in soup.find_all("a", href=lambda h: h and h.startswith("/library/")):
                        name_el = a.find("span", attrs={"x-test-search-response-title": True})
                        if not name_el:
                            continue
                        pulls_el = a.find("span", attrs={"x-test-pull-count": True})
                        name = name_el.get_text(strip=True)
                        sizes = [s.get_text(strip=True) for s in a.find_all("span", attrs={"x-test-size": True})]
                        caps  = [s.get_text(strip=True) for s in a.find_all("span", attrs={"x-test-capability": True})]
                        for size in (sizes or [""]):
                            match = re.match(r"(\d+\.?\d*)([bBmMkK])", size.split("x")[-1]) if size else None
                            if match:
                                val, unit = float(match.group(1)), match.group(2).lower()
                                p = val if unit == "b" else (val / 1e3 if unit == "m" else val / 1e6)
                            else:
                                continue
                            if p <= max_params_bil:
                                compatible.append({
                                    "Source": "Ollama",
                                    "Model": name,
                                    "Tag": size,
                                    "Params (B)": p,
                                    "Est. VRAM (GB)": round(p * bytes_per * disc_overhead, 1),
                                    "Capabilities": ", ".join(caps) or "—",
                                    "Pulls": pulls_el.get_text(strip=True) if pulls_el else "",
                                })
                except Exception:
                    continue
            prog.empty()

        if "HuggingFace" in disc_source:
            with st.spinner("Scanning HuggingFace top models…"):
                try:
                    hf_models = list(list_models(filter="text-generation", sort="downloads", direction=-1, limit=200))
                    for hf_m in hf_models:
                        nm = re.search(
                            r'(\d+\.?\d*)\s*[bB](?!\w)',
                            hf_m.id.replace("-", " ").replace("_", " ")
                        )
                        if nm:
                            p = float(nm.group(1))
                            if p <= max_params_bil:
                                compatible.append({
                                    "Source": "HuggingFace",
                                    "Model": hf_m.id,
                                    "Tag": f"{p}B",
                                    "Params (B)": p,
                                    "Est. VRAM (GB)": round(p * bytes_per * disc_overhead, 1),
                                    "Capabilities": "—",
                                    "Pulls": "",
                                })
                except Exception as e:
                    st.warning(f"HuggingFace scan failed: {e}")

        st.session_state["disc_results"] = compatible

    disc_results = st.session_state.get("disc_results")
    if disc_results is not None:
        if disc_results:
            df_compat = (
                pd.DataFrame(disc_results)
                .sort_values(["Source", "Params (B)"], ascending=[True, False])
                .reset_index(drop=True)
            )
            st.success(f"Found **{len(df_compat)}** compatible model variants for **{disc_vram} GB** VRAM at **{disc_prec}**")

            # Quick filter inside the results
            rf1, rf2 = st.columns(2)
            with rf1:
                src_filter = st.multiselect("Filter by source", df_compat["Source"].unique().tolist(), key="disc_src_filter")
            with rf2:
                cap_filter = st.multiselect("Filter by capability",
                    sorted(set(c for row in df_compat["Capabilities"] for c in row.split(", ") if c not in ("—", ""))),
                    key="disc_cap_filter")

            shown = df_compat
            if src_filter:
                shown = shown[shown["Source"].isin(src_filter)]
            if cap_filter:
                shown = shown[shown["Capabilities"].apply(lambda x: any(c in x for c in cap_filter))]

            st.dataframe(shown, use_container_width=True, hide_index=True)
            csv_disc = shown.to_csv(index=False)
            st.download_button("Export CSV", csv_disc, "compatible_models.csv", "text/csv", key="disc_csv")
        else:
            st.info(f"No models found that fit {disc_vram} GB at {disc_prec}. Try increasing VRAM or choosing a smaller precision.")

st.divider()
with st.expander("Assumptions & Notes"):
    st.markdown("""...""")  # Your original notes