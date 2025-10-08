from setuptools import setup

# Packaging the tutorial script as a single-module distribution.
# Users can 'pip install .' and get a console entry point 'globe-wind'.

setup(
    name="globe_wind",
    version="0.1.0",
    description="3D GDAS ARL wind viewer on an OpenGL globe (tutorial edition)",
    long_description=\"\"\"A tutorial-style, fully-commented wind visualization tool that reads
GDAS ARL (.w4) files and draws winds on a 3D globe using PyQt5 + PyOpenGL. Optional land mask
and Natural Earth coastlines included.\"\"\",\
    long_description_content_type="text/plain",
    author="Your Name",
    py_modules=["globe_wind_tutorial_full"],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "PyQt5",
        "PyOpenGL",
        "Pillow; python_version >= '3.9'",
        "requests; python_version >= '3.9'",
        # GeoPandas/Shapely are optional; include them if you want coastlines.
        # They can be heavy to build from source on some platforms.
        # "geopandas",
        # "shapely",
    ],
    entry_points={
        "console_scripts": [
            "globe-wind = globe_wind_tutorial_full:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
    ],
)
