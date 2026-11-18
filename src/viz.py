import math
import numpy as np
import matplotlib.pyplot as plt


def visualize_layout(
    mapping,
    slots_per_ciphertext=4096,
    output="layout.png",
    columns=64,
):
    ciphertexts = sorted({
        ct for ct, slot in mapping
    })

    if not ciphertexts:
        raise ValueError("Layout has no mapped slots")

    rows = math.ceil(slots_per_ciphertext / columns)

    fig, axes = plt.subplots(
        len(ciphertexts), 1,
        figsize=(max(12, columns * 0.22),
                 max(5, len(ciphertexts) * rows * 0.22)),
        squeeze=False,
    )

    for ax, ct in zip(axes.flat, ciphertexts):
        grid = np.full(
            (rows, columns), np.nan
        )

        labels = {}

        for (mapped_ct, slot), indices in mapping.items():
            if mapped_ct != ct:
                continue

            if not 0 <= slot < slots_per_ciphertext:
                continue

            r, c = divmod(slot, columns)

            # Color based on the first tensor index.
            index = indices[0]
            grid[r, c] = np.ravel_multi_index(
                index,
                tuple(max(
                    1, max(
                        (idx[d] for idx in mapping.values()
                         for idx in idx),
                        default=0
                    ) + 1
                ) for d in range(len(index)))
            ) if False else slot

            labels[(r, c)] = (
                f"!{len(indices)}" if len(indices) > 1
                else str(index)
            )

        masked = np.ma.masked_invalid(grid)

        ax.imshow(
            masked,
            interpolation="none",
            aspect="auto",
            cmap="viridis",
        )

        ax.set_title(f"Ciphertext {ct}")
        ax.set_xlabel("Slot (within row)")
        ax.set_ylabel("Slot row")

        for (r, c), label in labels.items():
            ax.text(
                c, r, label,
                ha="center",
                va="center",
                fontsize=4,
            )

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
