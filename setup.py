import os

from setuptools import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="kaggle-asl",
    version="0.0.1",
    author="Ayush Thakur",
    author_email="mein2work@gmail.com",
    description=(
        "Repository for Kaggle ASL Competition."
    ),
    license="Apache License Version 2.0",
    keywords="kaggle tensorflow keras",
    packages=["asl"],
    long_description=read("README.md"),
)
