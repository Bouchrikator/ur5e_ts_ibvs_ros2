from setuptools import setup

package_name = "sofa_ros2_adapter"

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
    description="SOFA Cosserat cable coupled to ROS 2 (TF boundary in, frames/markers out)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "cable_sofa_node = sofa_ros2_adapter.cable_sofa_node:main",
        ],
    },
)
