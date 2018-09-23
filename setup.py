from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(name="ordreg",
      version="0.1",
      description="Ordinal logistic regression in Python",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="http://github.com/dsaxton/ordreg",
      author="Daniel Saxton",
      author_email="daniel.saxton@gmail.com",
      license="BSD-3-Clause",
      packages=["ordreg"],
      install_requires=["numpy >= 1.14.0", "scipy >= 1.1.0", "scikit-learn >= 0.15.0"],
      zip_safe=False)
