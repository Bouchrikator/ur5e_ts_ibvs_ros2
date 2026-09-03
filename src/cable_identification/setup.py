from glob import glob

from setuptools import setup

package_name = "cable_identification"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/sofa",
         ["cable_identification/cable_scene.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Bouchrikator",
    maintainer_email="miko.boucherika@gmail.com",
    description="SOFA Cosserat cable model, configuration and identification pipeline",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cable_plugin_test = cable_identification.plugin_test:main",
            "cable_forward_test = cable_identification.forward_test:main",
            "optimus_smoke_test = cable_identification.optimus_smoke_test:main",
            "optimus_recovery_test = cable_identification.optimus_recovery_test:main",
            "optimus_pipeline_test = cable_identification.optimus_pipeline_test:main",
        ],
    },
)
