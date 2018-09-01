from setuptools import setup

setup(name="ordreg",
      version="0.1",
      description="Ordinal logistic regression in Python",
      url="http://github.com/dsaxton/ordreg",
      author="Daniel Saxton",
      author_email="daniel.saxton@gmail.com",
      license="BSD-3-Clause",
      packages=["ordreg"],
      install_requires=["numpy >= 1.14.0", "scipy >= 1.1.0"],
      zip_safe=False)
