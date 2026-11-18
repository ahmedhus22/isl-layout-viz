import islpy as isl
from core.models import LayoutEvaluation

def _coords(point, rank):
    result = []
    for d in range(rank):
        value = point.get_coordinate_val(isl.dim_type.set, d)
        if not value.is_int():
            raise ValueError("Non-integer coordinate encountered in ISL point")
        result.append(value.to_python())
    return tuple(result)

def evaluate_layout(isl_text):
    try:
        relation = isl.Map.read_from_str(isl.DEFAULT_CONTEXT, isl_text)
    except Exception as exc:
        raise ValueError(f"Invalid ISL map: {exc}") from exc

    domain_rank = relation.dim(isl.dim_type.in_)
    range_rank = relation.dim(isl.dim_type.out)
    if range_rank != 2:
        raise ValueError(f"Expected range [ct, slot] (2 dimensions), got {range_rank}")

    domain, slot_to_indices, index_to_slots = [], {}, {}

    def visit_domain(point):
        index = _coords(point, domain_rank)
        domain.append(index)
        singleton = isl.Set.from_point(point)
        image = relation.intersect_domain(singleton).range()

        def visit_image(image_point):
            ct, slot = _coords(image_point, 2)
            destination = (ct, slot)
            slot_to_indices.setdefault(destination, []).append(index)
            index_to_slots.setdefault(index, []).append(destination)

        image.foreach_point(visit_image)

    try:
        relation.domain().foreach_point(visit_domain)
    except Exception as exc:
        raise ValueError(
            "Could not enumerate this relation. Check that its domain and range "
            "are bounded and not excessively large."
        ) from exc

    domain.sort()
    bounds = []
    for d in range(domain_rank):
        values = [i[d] for i in domain]
        bounds.append((min(values), max(values))) if values else None

    return LayoutEvaluation(
        isl_text, domain_rank, slot_to_indices, index_to_slots, domain, bounds
    )
