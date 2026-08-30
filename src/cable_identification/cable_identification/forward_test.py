"""Phase 2 acceptance tests for the standalone Cosserat cable (headless).

  1. zero gravity + zero load  -> cable stays straight
  2. gravity                   -> repeatable tip sag
  3. EI x10                    -> smaller sag
  4. set_parameter + reinit    -> response changes mid-run
  5. frame-count refinement    -> sag insensitive to discretization
"""

import math
import sys


def _build(cfg_overrides=None, gravity=(0.0, 0.0, -9.81)):
    import Sofa.Core
    import Sofa.Simulation
    from cable_identification import cosserat_model as cm

    cfg = cm.load_config()
    cfg.update(cfg_overrides or {})
    cfg["gravity"] = list(gravity)
    root = Sofa.Core.Node("root")
    cm.prepare_root(root, cfg)
    cable = cm.build_cable(root, cfg)
    Sofa.Simulation.init(root)
    return root, cable


def _run(root, seconds):
    import Sofa.Simulation
    steps = int(round(seconds / root.dt.value))
    for _ in range(steps):
        Sofa.Simulation.animate(root, root.dt.value)


def _tip(cable):
    return list(cable.frame_poses()[-1][:3])


def main():
    results = []

    # 1: zero gravity -> straight
    root, cable = _build(gravity=(0.0, 0.0, 0.0))
    _run(root, 1.0)
    tip = _tip(cable)
    L = cable.cfg["length_m"]
    straight = abs(tip[0] - L) < 1e-4 and abs(tip[1]) < 1e-6 and abs(tip[2]) < 1e-6
    results.append(("zero-gravity straight", straight, f"tip={tip}"))

    # 2: gravity -> sag
    root, cable = _build()
    _run(root, 3.0)
    tip_g = _tip(cable)
    sag = -tip_g[2]
    results.append(("gravity sag > 5 mm", sag > 0.005, f"tip={tip_g} sag={sag:.4f} m"))

    # 3: EI x10 -> smaller sag
    root, cable = _build({"EI_Nm2": 0.1})
    _run(root, 3.0)
    sag_stiff = -_tip(cable)[2]
    results.append(("EI x10 reduces sag", sag_stiff < 0.5 * sag,
                    f"sag(EI x10)={sag_stiff:.4f} vs sag={sag:.4f}"))

    # 4: reinit mid-run changes response
    root, cable = _build()
    _run(root, 3.0)
    sag_before = -_tip(cable)[2]
    cable.set_parameter("EI", 0.1)
    _run(root, 3.0)
    sag_after = -_tip(cable)[2]
    results.append(("set_parameter+reinit takes effect", sag_after < 0.7 * sag_before,
                    f"before={sag_before:.4f} after={sag_after:.4f}"))

    # 5: discretization stability
    root, cable = _build({"number_of_sections": 24, "number_of_frames": 60})
    _run(root, 3.0)
    sag_fine = -_tip(cable)[2]
    rel = abs(sag_fine - sag) / max(sag, 1e-9)
    results.append(("sag stable under refinement (<10%)", rel < 0.10,
                    f"coarse={sag:.4f} fine={sag_fine:.4f} rel={100*rel:.1f}%"))

    # 6: table mode — tip bilaterally follows the grasp target (planar)
    root, cable = _build({"grasp_tip": True}, gravity=(0.0, 0.0, 0.0))
    L = cable.cfg["length_m"]
    target = [0.8 * L, 0.15 * L, 0.0, 0.0, 0.0, 0.0, 1.0]
    cable.set_grasp_pose(target)
    _run(root, 3.0)
    tip_t = _tip(cable)
    err = math.dist(tip_t, target[:3])
    base = list(cable.frame_poses()[0][:3])
    base_ok = math.dist(base, [0.0, 0.0, 0.0]) < 1e-3
    # 1 mm, not 5: above that the estimator cannot tell attachment compliance
    # apart from cable bending compliance.
    results.append(("grasped tip reaches target (<1 mm), base clamped",
                    err < 0.001 and base_ok,
                    f"tip={['%.3f' % v for v in tip_t]} err={err*1000:.2f} mm base_ok={base_ok}"))

    print()
    failed = 0
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failed += 0 if ok else 1
    print()
    if failed:
        print(f"CABLE_FORWARD_TEST_FAILED ({failed} failures)")
        return 1
    print("CABLE_FORWARD_TEST_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
