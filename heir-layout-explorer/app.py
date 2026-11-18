import json
import pandas as pd
import streamlit as st

from core.mlir_parser import extract_layouts
from core.isl_evaluator import evaluate_layout
from visualization.plots import slot_figure, tensor_figure

st.set_page_config(page_title="HEIR Layout Explorer", page_icon="▦", layout="wide")
st.title("HEIR Layout Explorer")
st.caption("Explore tensor-index → ciphertext-slot mappings using ISL evaluation.")

DEFAULT = r'''#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-64i0 - i1 + slot) mod 4096 = 0 and 0 <= i0 <= 63 and 0 <= i1 <= 63 and 0 <= slot <= 4095 }">

%arg0: tensor<1x!ct> { tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<64x64xf32>, layout = #layout> }
'''

with st.sidebar:
    st.header("Visualization")
    capacity = st.number_input("Slots per ciphertext", 1, 1_000_000, 4096, 256)
    columns = st.number_input("Slot grid columns", 1, 512, 64, 1)

mode = st.radio("Input", ["Paste MLIR", "Upload MLIR"], horizontal=True)
if mode == "Paste MLIR":
    mlir = st.text_area("MLIR module", value=DEFAULT, height=220)
else:
    upload = st.file_uploader("Upload .mlir file", type=["mlir", "txt"])
    mlir = upload.getvalue().decode("utf-8", errors="replace") if upload else ""

if not mlir.strip():
    st.info("Paste or upload an MLIR module to begin.")
    st.stop()

try:
    layouts = extract_layouts(mlir)
except Exception as exc:
    st.error(f"Layout extraction failed: {exc}")
    st.stop()

if not layouts:
    st.warning('No explicit #tensor_ext.layout<"..."> attributes found.')
    st.stop()

labels = [f"{x.name} · {x.source_kind} · {x.isl_text[:65]}" for x in layouts]
choice = st.selectbox("Extracted layout", range(len(layouts)),
                      format_func=lambda i: labels[i])
layout = layouts[choice]

with st.expander("ISL relation", expanded=False):
    st.code(layout.isl_text, language="text")

try:
    evaluated = evaluate_layout(layout.isl_text)
except Exception as exc:
    st.error(f"ISL evaluation failed: {exc}")
    st.exception(exc)
    st.stop()

stats = evaluated.statistics(int(capacity))
cols = st.columns(5)
for col, label, key in zip(
    cols,
    ["Tensor indices", "Mapping pairs", "Occupied slots", "Collisions", "Replicated indices"],
    ["domain_points", "mapping_pairs", "occupied_slots", "colliding_slots", "replicated_indices"]
):
    col.metric(label, f"{stats[key]:,}")

if stats["out_of_capacity_slots"]:
    st.warning(f"{stats['out_of_capacity_slots']} mapped slots exceed configured capacity {capacity}.")

slot_tab, tensor_tab, table_tab, diag_tab = st.tabs(
    ["Ciphertext slots", "Tensor view", "Mapping table", "Diagnostics"]
)

with slot_tab:
    cts = sorted({ct for ct, _ in evaluated.slot_to_indices})
    if cts:
        ct = st.selectbox("Ciphertext", cts, format_func=lambda x: f"Ciphertext {x}")
        st.plotly_chart(slot_figure(evaluated, ct, int(capacity), int(columns)),
                        use_container_width=True)
        st.caption("Green = occupied; orange = multiple tensor indices mapped to the same slot.")
    else:
        st.info("No integer points in this relation.")

with tensor_tab:
    if evaluated.domain_rank in (1, 2):
        st.plotly_chart(tensor_figure(evaluated), use_container_width=True)
        st.caption("Hover over an element to see its ciphertext/slot destination(s).")
    else:
        st.info(f"Tensor rank {evaluated.domain_rank}: use the mapping table or export.")

with table_tab:
    df = pd.DataFrame(evaluated.to_rows())
    if not df.empty:
        query = st.text_input("Filter mapping rows")
        if query:
            df = df[df.astype(str).apply(
                lambda c: c.str.contains(query, case=False, regex=False)
            ).any(axis=1)]
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode(),
                           "layout_mapping.csv", "text/csv")
    else:
        st.info("No mappings.")

with diag_tab:
    a, b, c = st.columns(3)
    a.metric("Unmapped tensor indices", stats["unmapped_indices"])
    b.metric("Indices with multiple destinations", stats["replicated_indices"])
    c.metric("Slots with multiple source indices", stats["colliding_slots"])
    if evaluated.replicated:
        st.subheader("Replicated tensor indices")
        st.dataframe(pd.DataFrame([
            {"tensor_index": str(i), "destination_count": n}
            for i, n in sorted(evaluated.replicated.items())
        ]), use_container_width=True, hide_index=True)
    if evaluated.collisions:
        st.subheader("Colliding slots")
        st.dataframe(pd.DataFrame([
            {"ciphertext": s[0], "slot": s[1], "tensor_indices": str(v)}
            for s, v in sorted(evaluated.collisions.items())
        ]), use_container_width=True, hide_index=True)
    if not evaluated.replicated and not evaluated.collisions:
        st.success("No replication or collisions detected.")
    payload = json.dumps({
        "layout": layout.name,
        "isl": layout.isl_text,
        "statistics": stats,
        "mappings": evaluated.to_rows(),
    }, indent=2)
    st.download_button("Download JSON analysis", payload,
                       "layout_analysis.json", "application/json")
