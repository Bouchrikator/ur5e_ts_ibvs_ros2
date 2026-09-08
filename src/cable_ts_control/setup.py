from glob import glob

from setuptools import setup

package_name = "cable_ts_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name, package_name + ".scripts"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bouchrikator",
    maintainer_email="miko.boucherika@gmail.com",
    description="Reduced cable state, Takagi-Sugeno PDC shape control and supervision",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cable_state_reducer_node = "
            "cable_ts_control.cable_state_reducer_node:main",
            "cable_ts_controller_node = "
            "cable_ts_control.cable_ts_controller_node:main",
            "cable_reference_projector_node = "
            "cable_ts_control.cable_reference_projector_node:main",
            "cable_supervisor_node = "
            "cable_ts_control.cable_supervisor_node:main",
            "cable_metrics_recorder_node = "
            "cable_ts_control.cable_metrics_recorder_node:main",
            "cable_modal_observer_node = "
            "cable_ts_control.cable_modal_observer_node:main",
            "generate_sofa_dataset = "
            "cable_ts_control.scripts.generate_sofa_dataset:main",
            "cable_sofa_mor_pipeline = cable_ts_control.sofa_mor_pipeline:main",
            "validate_cosserat_rom = cable_ts_control.scripts.validate_cosserat_rom:main",
            "build_modal_basis = "
            "cable_ts_control.scripts.build_modal_basis:main",
            "identify_ts_vertices = "
            "cable_ts_control.scripts.identify_ts_vertices:main",
            "solve_cable_ts_lmi = "
            "cable_ts_control.scripts.solve_cable_ts_lmi:main",
            "settle_shape_jacobians = "
            "cable_ts_control.scripts.settle_shape_jacobians:main",
        ],
    },
)
