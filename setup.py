from setuptools import setup, find_packages

DESCRIPTION = "Image detection for Robot-Framework"
LONG_DESCRIPTION = """
Supercharge your Robot Framework automation with our advanced image detection package. Seamlessly integrate machine learning capabilities to train datasets and efficiently detect images in your tests.

Key Features:
- Utilize the power of machine learning for image detection tasks
- Train custom datasets to improve accuracy and recognition
- Simplify Robot Framework test automation with enhanced image recognition

Take your Robot Framework tests to the next level with our image detection package.
"""

setup(
    name="robotframework-imagedetection",
    version="1.0.0",  # Replace with your desired version number
    author="M.Kherki(Alpha-Centauri-00)",
    author_email="alpha_Centauri@posteo.de",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/Alpha-Centauri-00/robotframework-imagedetection",
    packages=find_packages(),
    install_requires=[],  # Add your dependencies here
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
