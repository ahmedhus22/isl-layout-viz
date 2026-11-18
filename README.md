# HEIR Layout Explorer

Browser-based MVP for extracting `tensor_ext.layout` attributes from MLIR,
evaluating their ISL relations, and inspecting tensor-to-ciphertext-slot mappings.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Paste an MLIR module or upload an `.mlir` file. The app extracts explicit
`#tensor_ext.layout<"...">` attributes and evaluates the relation with `islpy`.

## Current scope

- Explicit serialized `tensor_ext.layout` attributes (including inline ones).
- ISL maps with a two-dimensional range `[ct, slot]`.
- Bounded relations that are practical to enumerate.
- Tensor heatmaps for rank-1 and rank-2 domains.
- Ciphertext slot occupancy, mapping table, collision/replication diagnostics.
- CSV and JSON export.

This is a focused text extractor, not a complete MLIR parser. It does not yet
resolve every possible MLIR alias/attribute form or visualize rank > 2 tensors.
