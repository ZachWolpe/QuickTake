import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quicktake",
    version="0.0.10",
    author="Zach Wolpe",
    description="Off-the-shelf computer vision ML models. Yolov5, gender and age determination.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    py_modules=["quicktake"],
    package_dir={'':"quicktake/src"},
    install_requires=[]
)