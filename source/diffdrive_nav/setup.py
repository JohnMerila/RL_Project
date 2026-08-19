"""Install the diffdrive_nav Isaac Lab extension as a Python package."""

from pathlib import Path

from setuptools import find_packages, setup

PACKAGE_ROOT = Path(__file__).parent

setup(
    name="diffdrive_nav",
    version="0.1.0",
    description="Isaac Lab LiDAR point-goal navigation task for differential-drive robots",
    packages=find_packages(),
    include_package_data=True,
    package_data={"diffdrive_nav": ["assets/*.urdf"]},
    python_requires=">=3.10",
    zip_safe=False,
)

