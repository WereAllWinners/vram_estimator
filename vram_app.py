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

st.set_page_config(page_title="VRAM Estimator", page_icon="🧠", layout="wide")

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
    per_gpu_vram = st.number_input(f"VRAM per GPU ({unit_label.split()[0]})", min_value=0.0, value=48.0, step=1.0)
    available_vram = per_gpu_vram * num_gpus  # Backend math for total

    parallelism_type = st.selectbox("Strategy", ["None", "Tensor Parallel", "Pipeline Parallel", "Data Parallel"])

    benchmark = st.checkbox("Run Benchmark for Activations (Slow)")

    if st.button("Detect Hardware"):
        gpus = GPUtil.getGPUs()
        if gpus:
            detected_per_gpu = gpus[0].memoryTotal / 1024 if unit_label.startswith("GiB") else gpus[0].memoryTotal * (1024**3 / 1e9) / 1024
            per_gpu_vram = detected_per_gpu  # Update per-GPU input
            available_vram = per_gpu_vram * num_gpus  # Recompute total
            st.write(f"Detected per GPU: {per_gpu_vram:.2f} {unit_label.split()[0]} (Total: {available_vram:.2f})")
        else:
            st.warning("No GPU detected.")

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
with st.expander("Assumptions & Notes"):
    st.markdown("""...""")  # Your original notes