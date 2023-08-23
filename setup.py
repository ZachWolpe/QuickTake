import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()




setuptools.setup(
    name                            = "quicktake",
    version                         = "0.0.10",
    author                          = "Zach Wolpe",
    author_email                    = "zachcolinwolpe@gmail.com",
    description                     = "Off-the-shelf computer vision ML models. Yolov5, gender and age determination.",
    long_description                = long_description,
    long_description_content_type   = "text/markdown",
    packages                        = setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires                 = '>=3.8',
    py_modules                      = ["quicktake"],
    package_dir                     = {'':"quicktake/src"},
    install_requires                =[
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "torchviz>=0.0.2",
        "tqdm>=4.66.1",
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "opencv-python>=4.8.0",
        "plotly>=5.16.1",
        "plotly-express>=0.4.1",
        "matplotlib>=3.7.2",
        "pillow>=10.0.0"
        ]
)