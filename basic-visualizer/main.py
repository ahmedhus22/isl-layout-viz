from eval import evaluate_layout
from viz import visualize_layout

layout = """
{
  [i0, i1] -> [ct, slot] :
  ct = 0
  and (-64i0 - i1 + slot) mod 4096 = 0
  and 0 <= i0 <= 63
  and 0 <= i1 <= 63
  and 0 <= slot <= 4095
}
"""

mapping, stats = evaluate_layout(layout)

print(stats)

visualize_layout(
    mapping,
    slots_per_ciphertext=4096,
    output="layout.png",
)