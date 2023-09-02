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

with open(join(CURDIR, "ImageDetection", "Base_detection.py"), encoding="utf-8") as f:
    VERSION = re.search(r'\n__version__ = "(.*)"', f.read()).group(1)

with open("requirements.txt") as f:
    _required_packages = f.read().splitlines()

setup(
    name="robotframework-imagedetection",
    version=VERSION,
    author="M.Kherki(Alpha-Centauri-00)",
    author_email="alpha_Centauri@posteo.de",
    description=DESCRIPTION,
    long_description=Long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/Alpha-Centauri-00/robotframework-imagedetection",
    packages=["Imagedetection"],
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
)
