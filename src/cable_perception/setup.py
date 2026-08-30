from setuptools import setup

package_name = "cable_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bouchrikator",
    maintainer_email="miko.boucherika@gmail.com",
    description="Cable marker observation sources (synthetic and eye-to-hand camera)",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "synthetic_marker_node = "
            "cable_perception.synthetic_marker_node:main",
            "cable_marker_tracker_node = "
            "cable_perception.cable_marker_tracker_node:main",
            "cable_dlo_detector_node = "
            "cable_perception.cable_dlo_detector_node:main",
        ],
    },
)
