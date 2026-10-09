"""Headless gate for the pinned official MOR mapping, Vec1d -> Vec3d and Vec1d -> Vec6d."""

import os
from pathlib import Path
import sys
import tempfile

import numpy as np


MOR_COMMIT = "d94dc49dff66d936ad11c33a8f195cc167c98b61"


def check_mapping():
    prefix = Path(os.environ.get("SOFA_ROOT", "/opt/sofa")) / "plugins/ModelOrderReduction"
    marker = prefix / ".cable-mor-build"
    if not marker.is_file() or marker.read_text().split()[0] != MOR_COMMIT:
        raise RuntimeError("Install the pinned plugin with install_model_order_reduction.sh")
    for template in ("Vec3d", "Vec6d"):
        check_mapping_template(template)
    print(f"MOR revision: {MOR_COMMIT}")
    print("Position, velocity, transpose force and assembled mapped mass (Vec3d, Vec6d): passed")


def check_mapping_template(template):
    """Two modes over two output nodes of ``dim`` components; every operation is checked
    against numpy, so a leftover three-component assumption fails for Vec6d."""
    import Sofa.Core
    import Sofa.Simulation

    dim = int(template[3])
    rng = np.random.default_rng(dim)
    modes = rng.normal(size=(2 * dim, 2))
    neutral = rng.normal(size=(2, dim))
    initial = np.array([0.1, -0.2])
    velocity = np.array([0.02, -0.01])
    forces = rng.normal(size=(2, dim)) * 0.05
    timestep = 0.01
    with tempfile.TemporaryDirectory(prefix="cable-mor-plugin-") as directory:
        modes_path = Path(directory) / "modes.txt"
        np.savetxt(modes_path, modes, header=f"{2 * dim} 2", comments="", fmt="%.17g")
        root = Sofa.Core.Node("mor_mapping_gate")
        root.gravity = [0.] * 3
        root.dt = timestep
        root.addObject("RequiredPlugin", pluginName=[
            "Sofa.Component.AnimationLoop", "Sofa.Component.StateContainer",
            "Sofa.Component.ODESolver.Backward", "Sofa.Component.LinearSolver.Direct",
            "Sofa.Component.Mass", "Sofa.Component.MechanicalLoad",
            "ModelOrderReduction"])
        root.addObject("DefaultAnimationLoop")
        root.addObject("EulerImplicitSolver", rayleighMass=0., rayleighStiffness=0.)
        root.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixd")
        modal = root.addChild("modalCoordinate")
        modal_mo = modal.addObject("MechanicalObject", template="Vec1d",
                                   position=initial.tolist(), velocity=velocity.tolist())
        full = modal.addChild("fullCoordinate")
        full_mo = full.addObject("MechanicalObject", template=template,
                                 position=neutral.tolist(), rest_position=neutral.tolist())
        mapping = full.addObject(
            "ModelOrderReductionMapping", name="strainModalMapping",
            input=modal_mo.getLinkPath(), output=full_mo.getLinkPath(),
            modesPath=str(modes_path.resolve()))
        full.addObject("UniformMass", totalMass=2.)
        full.addObject("ConstantForceField", template=template, forces=forces.tolist())
        try:
            Sofa.Simulation.init(root)
            assert mapping.getClassName() == "ModelOrderReductionMapping"
            np.testing.assert_allclose(full_mo.position.value.ravel(),
                                       neutral.ravel() + modes @ initial, atol=1e-12)
            Sofa.Simulation.animate(root, timestep)
            # UniformMass(totalMass=2) over two nodes: M = I per component, M_r = Phi^T Phi
            expected_velocity = velocity + timestep * np.linalg.solve(
                modes.T @ modes, modes.T @ forces.ravel())
            np.testing.assert_allclose(modal_mo.velocity.value.ravel(),
                                       expected_velocity, atol=1e-12)
            np.testing.assert_allclose(full_mo.velocity.value.ravel(),
                                       modes @ expected_velocity, atol=1e-12)
            np.testing.assert_allclose(full_mo.position.value.ravel(),
                                       neutral.ravel() + modes @ (
                                           initial + timestep * expected_velocity), atol=1e-12)
        finally:
            Sofa.Simulation.unload(root)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cable-modes")
    parser.add_argument("--cable-metadata")
    args = parser.parse_args()
    try:
        check_mapping()
        if args.cable_modes or args.cable_metadata:
            check_cable(args.cable_modes, args.cable_metadata)
    except Exception as error:
        print(f"CABLE_SOFA_MOR_PLUGIN_TEST_FAILED: {error}", file=sys.stderr)
        return 1
    print("CABLE_SOFA_MOR_PLUGIN_TEST_PASSED")
    return 0


def check_cable(modes_path, metadata_path):
    import Sofa.Core
    import Sofa.Simulation
    import yaml
    from ament_index_python.packages import get_package_share_directory
    from cable_identification import cosserat_model as cm
    from cable_identification.strain_basis import ReductionSpec

    cfg = cm.load_config(str(Path(get_package_share_directory("cable_identification"))
                             / "config/cable_truth.yaml"))
    with open(metadata_path) as stream:
        metadata = yaml.safe_load(stream)
    for reduction in (None, ReductionSpec(modes_path, metadata_path, metadata["n_modes"])):
        root = Sofa.Core.Node("cable_restore_gate")
        cm.prepare_root(root, cfg, extra_plugins=[] if reduction is None else ["ModelOrderReduction"])
        cable = cm.build_cable(root, cfg, reduction=reduction)
        try:
            Sofa.Simulation.init(root)
            assert (cable.modal_mo is None) == (reduction is None)
            assert cable.mapping.getClassName() == "DiscreteCosseratMapping"
            for _ in range(20):
                Sofa.Simulation.animate(root, cfg["timestep_s"])
            saved = cable.save_state()
            markers = np.asarray(cable.marker_positions())
            independent = cable.strain_mo if reduction is None else cable.modal_mo
            with independent.position.writeable() as position:
                position[:] += 0.001
            cable.restore_state(saved)
            if reduction is None:
                cable.refresh_mapping()
            np.testing.assert_allclose(cable.marker_positions(), markers, atol=1e-12, rtol=0.)
            for key, value in saved.items():
                np.testing.assert_array_equal(cable.save_state()[key], value)
        finally:
            Sofa.Simulation.unload(root)
    print("Default FOM, optional ROM and independent-state restoration: passed")


if __name__ == "__main__":
    sys.exit(main())