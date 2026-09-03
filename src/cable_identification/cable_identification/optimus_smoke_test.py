"""Optimus compatibility gate.

Step 5 of the plan. Optimus has no binary release for SOFA v25.12, so before
connecting it to the Cosserat cable we check, in this order:

  1. the plugin loads at all;
  2. every component the estimator needs can be instantiated;
  3. what data fields each of them actually exposes.

Nothing here touches the cable. If this test does not pass, the estimator must
stay on its fallback and the identification story is simply not ready yet.

    ros2 run cable_identification optimus_smoke_test
"""

import sys

# The seven components of the ported Optimus core (docs/optimus_port.md), i.e.
# everything the ROUKF parameter estimator needs. UKFilterClassic is not ported.
REQUIRED_COMPONENTS = [
    "OptimParams",
    "FilteringAnimationLoop",
    "StochasticStateWrapper",
    "MappedStateObservationManager",
    "SimulatedStateObservationSource",
    "PreStochasticWrapper",
    "ROUKFilter",
]


def check_plugin_loads():
    import SofaRuntime

    loaded = SofaRuntime.importPlugin("Optimus")
    print(f"importPlugin('Optimus'): {loaded}")
    return bool(loaded)


def check_components():
    """Instantiate every required component and report its data fields."""
    import Sofa.Core

    root = Sofa.Core.Node("root")
    missing, available = [], {}

    for name in REQUIRED_COMPONENTS:
        try:
            obj = root.addObject(name, name=f"probe_{name}")
        except Exception as error:
            missing.append(name)
            print(f"  MISSING  {name}: {type(error).__name__}")
            continue
        fields = sorted(d.getName() for d in obj.getDataFields())
        available[name] = fields
        print(f"  OK       {name}")
        print(f"           data fields: {', '.join(fields)}")

    return missing, available


def main():
    print("=" * 70)
    print("OPTIMUS COMPATIBILITY GATE")
    print("=" * 70)

    try:
        loaded = check_plugin_loads()
    except ImportError as error:
        print(f"SofaPython3 unavailable: {error}")
        print("OPTIMUS_SMOKE_TEST_FAILED")
        return 1

    if not loaded:
        print("\nOptimus is not installed in this image.")
        print("Rebuild with: docker compose build --build-arg WITH_OPTIMUS=true")
        print("OPTIMUS_SMOKE_TEST_FAILED")
        return 1

    print("\ninstantiating the required components:")
    missing, _ = check_components()

    if missing:
        print(f"\n{len(missing)} component(s) unavailable: {', '.join(missing)}")
        print("Do NOT wire Optimus to Cosserat until these resolve.")
        print("OPTIMUS_SMOKE_TEST_FAILED")
        return 1

    print("\nall required components instantiate against this SOFA build")
    print("OPTIMUS_SMOKE_TEST_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
