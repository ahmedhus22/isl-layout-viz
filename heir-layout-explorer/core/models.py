from dataclasses import dataclass, field

@dataclass
class LayoutSpec:
    name: str
    isl_text: str
    source_kind: str
    start_offset: int
    end_offset: int

@dataclass
class LayoutEvaluation:
    isl_text: str
    domain_rank: int
    slot_to_indices: dict = field(default_factory=dict)
    index_to_slots: dict = field(default_factory=dict)
    domain_indices: list = field(default_factory=list)
    domain_bounds: list = field(default_factory=list)

    @property
    def replicated(self):
        return {i: len(s) for i, s in self.index_to_slots.items() if len(s) > 1}

    @property
    def collisions(self):
        return {s: i for s, i in self.slot_to_indices.items() if len(i) > 1}

    def statistics(self, capacity=4096):
        return {
            "domain_points": len(self.domain_indices),
            "mapping_pairs": sum(map(len, self.index_to_slots.values())),
            "occupied_slots": len(self.slot_to_indices),
            "ciphertext_count": len({ct for ct, _ in self.slot_to_indices}),
            "colliding_slots": len(self.collisions),
            "replicated_indices": len(self.replicated),
            "unmapped_indices": sum(not self.index_to_slots.get(i) for i in self.domain_indices),
            "out_of_capacity_slots": sum(slot < 0 or slot >= capacity for _, slot in self.slot_to_indices),
        }

    def to_rows(self):
        rows = []
        for index in sorted(self.index_to_slots):
            destinations = sorted(self.index_to_slots[index])
            for ct, slot in destinations:
                rows.append({
                    "tensor_index": str(index),
                    "ciphertext": ct,
                    "slot": slot,
                    "destinations_for_index": len(destinations),
                    "sources_for_slot": len(self.slot_to_indices[(ct, slot)]),
                })
        return rows
