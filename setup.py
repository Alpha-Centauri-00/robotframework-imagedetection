import re
from setuptools import setup, find_packages
from os.path import abspath, dirname, join

DESCRIPTION = "Image detection for Robot-Framework"
LONG_DESCRIPTION = """
Supercharge your Robot Framework automation with our advanced image detection package. Seamlessly integrate machine learning capabilities to train datasets and efficiently detect images in your tests.

Key Features:
- Utilize the power of machine learning for image detection tasks
- Train custom datasets to improve accuracy and recognition
- Simplify Robot Framework test automation with enhanced image recognition

Take your Robot Framework tests to the next level with our image detection package.

"""

CURDIR = dirname(abspath(__file__))
with open("README.md", "r", encoding="utf-8") as fh:
    Long_desc = LONG_DESCRIPTION + fh.read()

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    _required_packages = f.readlines()

# Remove newline characters from the end of each line
_required_packages = [line.strip() for line in _required_packages]

setup(
    name="robotframework-imagedetection",
    use_scm_version=True,  # Use setuptools-scm to determine version
    author="M.Kherki(Alpha-Centauri-00)",
    author_email="alpha_Centauri@posteo.de",
    description=DESCRIPTION,
    long_description=Long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/Alpha-Centauri-00/robotframework-imagedetection",
    packages=find_packages(),
    install_requires=_required_packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Testing",
        "Topic :: Software Development :: Testing :: Acceptance",
        "Framework :: Robot Framework",
    ],
    python_requires=">=3.6",
    setup_requires=["setuptools-scm"],  # Add setuptools-scm as a setup requirement
)
