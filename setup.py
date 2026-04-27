# PEIRASTIC_control: standalone Franka control (Python peirastic package)
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
readme_path = path.join(this_directory, "README.md")
long_description = ""
if path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        lines = [x for x in f.readlines() if ".png" not in x]
        long_description = "".join(lines)

setup(
    name="peirastic",
    packages=[package for package in find_packages() if package.startswith("peirastic")],
    package_data={
        "peirastic.netft_calib": [
            "config/*.txt",
            "config/*.yml",
        ],
    },
    install_requires=[
        "pyzmq",
        "numpy",
        "PyYAML",
        "easydict",
        "termcolor",
        "scipy",
        "protobuf>=3.20.0,<3.21.0",
        "hidapi>=0.14",
        "glfw",
        "matplotlib",
        "rospkg",
        "netifaces>=0.11.0",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="Franka Panda control via ZMQ (PEIRASTIC_control standalone)",
    author="PEIRASTIC",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        'console_scripts': [
            'peirastic.get_controller_list=peirastic.scripts.get_peirastic_info:default_controller_list',
            'peirastic.get_controller_info=peirastic.scripts.get_peirastic_info:default_all_controller_info',
            'peirastic.reset_joints=peirastic.scripts.reset_robot_joints:main',
            'peirastic.print_robot_state=peirastic.scripts.print_robot_state:main',
            'peirastic.spacenav_teleop=peirastic.scripts.spacenav_teleop_publisher:main',
            'peirastic.spacenav_cartesian_min=peirastic.scripts.spacenav_cartesian_min:main',
            'peirastic.spacenav_mode_switch_test=peirastic.scripts.spacenav_mode_switch_test:main',
            'peirastic.netft_data_acquisition=peirastic.scripts.netft_data_acquisition:main',
            'peirastic.netft_identification=peirastic.scripts.netft_identification:main',
            'peirastic.netft_pub_calib_result=peirastic.scripts.netft_pub_calib_result:main',
        ]
    },
)
