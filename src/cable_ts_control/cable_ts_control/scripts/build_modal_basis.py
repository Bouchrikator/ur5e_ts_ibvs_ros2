#!/usr/bin/env python3
"""Build the modal basis of the cable from a SOFA dataset.

Step 6 of the plan: PCA over the recorded shapes, keeping the dominant modes.

    ros2 run cable_ts_control build_modal_basis \
        --dataset /tmp/cable_dataset.npz --output cable_modal_basis.yaml
"""

import argparse
import sys

import numpy as np

from cable_ts_control.modal_basis import (
    ModalBasis,
    build_modal_basis,
    explained_variance_ratio,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="dataset .npz")
    parser.add_argument("--output", required=True, help="basis YAML to write")
    parser.add_argument("--modes", type=int, default=2)
    parser.add_argument("--min-explained", type=float, default=0.9,
                        help="fail if the kept modes explain less than this")
    args = parser.parse_args(argv)

    data = np.load(args.dataset, allow_pickle=False)
    shapes = data["shapes"]

    mean, phi, singular_values = build_modal_basis(shapes, args.modes)
    explained = explained_variance_ratio(singular_values, args.modes)

    print(f"dataset: {shapes.shape[0]} shapes of dimension {shapes.shape[1]}")
    for index in range(min(6, len(singular_values))):
        share = explained_variance_ratio(singular_values, index + 1)
        print(f"  mode {index + 1}: sigma={singular_values[index]:.4e}, "
              f"cumulative energy={share * 100:.2f}%")

    if explained < args.min_explained:
        print(f"ERROR: {args.modes} modes explain only {explained * 100:.2f}%, "
              f"below the {args.min_explained * 100:.0f}% requirement")
        return 1

    basis = ModalBasis(mean, phi, data["marker_s_over_l"].tolist(), singular_values)
    basis.save(args.output)

    q = np.array([basis.project(shape) for shape in shapes])
    rmse = float(np.sqrt(np.mean(
        (shapes - np.array([basis.reconstruct(v) for v in q])) ** 2)))
    print(f"\nkept {args.modes} modes, {explained * 100:.2f}% of the energy")
    print(f"reconstruction rmse: {rmse * 1e3:.3f} mm")
    for index in range(args.modes):
        print(f"  q{index + 1} range: [{q[:, index].min():+.4f}, "
              f"{q[:, index].max():+.4f}]")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
