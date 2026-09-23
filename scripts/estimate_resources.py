"""Conservative in-memory planning estimates, not measured peak memory."""
from __future__ import annotations
import argparse
import json
import math


def estimate(train_count=50000, val_count=10000, image_size=32, max_particles=200, neighbors=6):
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0
           for x in [train_count, val_count, image_size, max_particles, neighbors]):
        raise ValueError('Resource dimensions must be positive integers')
    fitting = train_count+val_count
    particle_bytes = fitting*max_particles*4*8
    image_bytes = fitting*image_size**2*4
    node_bytes = fitting*max_particles*4*4
    adjacency_bytes = fitting*max_particles*(2*neighbors+1)*16
    # Account for coexisting source/working arrays and a 1 GiB runtime allowance.
    estimate_bytes = 2*(particle_bytes+image_bytes+node_bytes+adjacency_bytes)+2**30
    return {'train_jets': train_count, 'validation_jets': val_count,
            'assumed_constituents_per_jet': max_particles,
            'components_bytes': {'double_four_vectors': particle_bytes, 'single_images': image_bytes,
                                 'single_node_features': node_bytes, 'sparse_graph_allowance': adjacency_bytes},
            'planning_peak_gib': estimate_bytes/2**30,
            'fitting_particle_array_bytes': fitting*max_particles*4*4,
            'mat_v5_array_limit_ok': fitting*max_particles*4*4 < 2**31,
            'interpretation': 'Conservative planning estimate using 200 particles per jet; not measured RAM, an exact upper bound, or a guarantee. Excludes the optional reference model.'}


def enforce_budget(plan, budget_gib):
    if not plan['mat_v5_array_limit_ok']:
        raise ValueError('Fitting array exceeds the current MAT-v5 limit; use a streamed/HDF5 trainer before this scale')
    if budget_gib is not None and (not math.isfinite(budget_gib) or budget_gib <= 0 or plan['planning_peak_gib'] > budget_gib):
        raise ValueError(f"Planning estimate {plan['planning_peak_gib']:.2f} GiB exceeds the supplied budget")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-count', type=int, default=50000)
    parser.add_argument('--val-count', type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(estimate(args.train_count, args.val_count), indent=2))
