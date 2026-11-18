import islpy as isl


def evaluate_layout(layout_str):
    """
    Evaluate an ISL tensor-index -> [ct, slot] relation.

    Returns:
        mapping: {(ct, slot): [tensor_index, ...]}
        stats:   replication and collision statistics
    """
    relation = isl.Map.read_from_str(
        isl.DEFAULT_CONTEXT, layout_str
    )

    domain_rank = relation.dim(isl.dim_type.in_)
    range_rank = relation.dim(isl.dim_type.out)

    if range_rank != 2:
        raise ValueError(
            f"Expected [ct, slot], got {range_rank} dimensions"
        )

    mapping = {}
    source_counts = {}

    def get_coords(point, dim_type, rank):
        return tuple(
            point.get_coordinate_val(
                dim_type, i
            ).to_python()
            for i in range(rank)
        )

    def visit_domain(point):
        tensor_idx = get_coords(
            point, isl.dim_type.set, domain_rank
        )

        source_counts[tensor_idx] = 0

        singleton = isl.Set.from_point(point)

        # Evaluate all range points corresponding to this
        # particular tensor index.
        ranges = (
            relation
            .intersect_domain(singleton)
            .range()
        )

        def visit_range(range_point):
            ct, slot = get_coords(
                range_point, isl.dim_type.set, 2
            )

            mapping.setdefault((ct, slot), []).append(
                tensor_idx
            )

            source_counts[tensor_idx] += 1

        ranges.foreach_point(visit_range)

    relation.domain().foreach_point(visit_domain)

    replications = {
        idx: count
        for idx, count in source_counts.items()
        if count > 1
    }

    collisions = {
        location: indices
        for location, indices in mapping.items()
        if len(indices) > 1
    }

    stats = {
        "tensor_elements": len(source_counts),
        "mapped_elements": sum(source_counts.values()),
        "occupied_slots": len(mapping),
        "replicated_elements": len(replications),
        "colliding_slots": len(collisions),
    }

    return mapping, stats
