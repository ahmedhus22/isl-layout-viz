import math
import numpy as np
import plotly.graph_objects as go

def slot_figure(result, ct, capacity, columns):
    columns = max(1, int(columns))
    rows = max(1, math.ceil(capacity / columns))
    z = np.full((rows, columns), np.nan)
    hover = np.full((rows, columns), "", dtype=object)

    for (ciphertext, slot), indices in result.slot_to_indices.items():
        if ciphertext != ct or not (0 <= slot < capacity):
            continue
        r, c = divmod(slot, columns)
        z[r, c] = 2 if len(indices) > 1 else 1
        hover[r, c] = f"Ciphertext: {ct}<br>Slot: {slot}<br>Tensor index/indices: {indices}"

    fig = go.Figure(go.Heatmap(
        z=z, customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        colorscale=[[0, "#edf0f4"], [0.49, "#edf0f4"],
                    [0.5, "#35a879"], [0.74, "#35a879"],
                    [0.75, "#e29b35"], [1, "#e29b35"]],
        zmin=0, zmax=2, xgap=1, ygap=1,
        colorbar=dict(title="Occupancy", tickvals=[1, 2],
                      ticktext=["Occupied", "Collision"])
    ))
    fig.update_layout(
        title=f"Ciphertext {ct}: slot occupancy",
        xaxis_title="Slot column", yaxis_title="Slot row",
        height=min(850, max(420, rows * 13)),
        margin=dict(l=45, r=20, t=55, b=45)
    )
    fig.update_yaxes(autorange="reversed")
    return fig

def tensor_figure(result):
    rank = result.domain_rank
    if rank not in (1, 2):
        raise ValueError("Tensor heatmap currently supports rank 1 and 2")
    if not result.domain_indices:
        return go.Figure()

    if rank == 1:
        lo, hi = result.domain_bounds[0]
        shape = (hi - lo + 1, 1)
        indices = [(i,) for i in range(lo, hi + 1)]
    else:
        (lo0, hi0), (lo1, hi1) = result.domain_bounds
        shape = (hi0 - lo0 + 1, hi1 - lo1 + 1)
        indices = [(i, j) for i in range(lo0, hi0 + 1)
                   for j in range(lo1, hi1 + 1)]

    z = np.full(shape, np.nan)
    hover = np.full(shape, "", dtype=object)
    for index in indices:
        destinations = result.index_to_slots.get(index, [])
        if not destinations:
            continue
        r = index[0] - result.domain_bounds[0][0]
        c = 0 if rank == 1 else index[1] - result.domain_bounds[1][0]
        z[r, c] = min(len(destinations), 3)
        hover[r, c] = f"Tensor index: {index}<br>Destination(s): {sorted(destinations)}"

    fig = go.Figure(go.Heatmap(
        z=z, customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        colorscale=[[0, "#edf0f4"], [0.32, "#edf0f4"],
                    [0.33, "#5b8fd1"], [0.65, "#5b8fd1"],
                    [0.66, "#e29b35"], [1, "#e29b35"]],
        zmin=0, zmax=3, xgap=1, ygap=1,
        colorbar=dict(title="Destinations", tickvals=[1, 2, 3],
                      ticktext=["1", "2", "3+"])
    ))
    fig.update_layout(
        title="Tensor index → ciphertext slot",
        xaxis_title="Tensor dimension 1" if rank == 2 else "",
        yaxis_title="Tensor dimension 0",
        height=min(850, max(420, shape[0] * 12)),
        margin=dict(l=55, r=20, t=55, b=45)
    )
    fig.update_yaxes(autorange="reversed")
    return fig
