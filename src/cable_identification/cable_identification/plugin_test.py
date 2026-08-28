"""Phase 1 smoke test: Cosserat plugin loads and key components instantiate."""

import sys


def main():
    import Sofa
    import Sofa.Core
    import Sofa.Simulation
    import SofaRuntime

    from cable_identification import cosserat_model as cm

    ok = SofaRuntime.importPlugin("Cosserat")
    print(f"importPlugin('Cosserat'): {ok}")

    root = Sofa.Core.Node("root")
    cfg = cm.load_config()
    cm.prepare_root(root, cfg)
    cable = cm.build_cable(root, cfg)
    Sofa.Simulation.init(root)

    fields = sorted(d.getName() for d in cable.force_field.getDataFields())
    required = ["EI", "GI", "EA", "GA", "useInertiaParams", "rayleighStiffness"]
    missing = [f for f in required if f not in fields]
    print("BeamHookeLawForceField fields:", fields)
    print("EI =", cable.force_field.EI.value, " GI(=GJ) =", cable.force_field.GI.value)

    n_frames = len(cable.frame_poses())
    print(f"centerline frames: {n_frames}, markers at indices {cable.marker_indices}")

    if missing:
        print(f"FAIL: missing fields {missing}")
        return 1
    print("CABLE_PLUGIN_TEST_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
