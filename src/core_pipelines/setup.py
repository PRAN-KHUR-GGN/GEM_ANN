# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
from setuptools import find_packages, setup

# get the dependencies and installs
with open("requirements.txt", "r", encoding="utf-8") as f:
    requires = []
    for line in f:
        req = line.split("#", 1)[0].strip()
        if req and not req.startswith("--"):
            requires.append(req)

# get test dependencies and installs
with open("test_requirements.txt", "r", encoding="utf-8") as f:
    test_requires = [x.strip() for x in f if x.strip() and not x.startswith("-r")]


extras_require = {
    "pyspark": ["pyspark>3.0", "chispa~=0.6.0", "py4j"],
    "mlflow": ["mlflow~=1.0"],
    "shap": ["shap>=0.37"],
    "ray_tf": [
        "tensorflow>=2.7.0, <2.8.0",
        "tensorflow_lattice>=2.0.9",
        "tensorflow_probability>=0.12, <1.0.0",
        "ray[tune]>=1.8, <2.0.0",
    ],
}

setup(
    name="core_pipelines",
    version="0.7.20",
    description="Generic Pipelines",
    author="QuantumBlack Labs",
    author_email="feedback@quantumblack.com",
    packages=find_packages(exclude=["docs*", "tests*", "tools*"]),
    include_package_data=True,
    package_data={},
    tests_require=test_requires,
    install_requires=requires,
    extras_require=extras_require,
)
