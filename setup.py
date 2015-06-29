import os
from setuptools import find_packages, setup


def main():
    def read(fname):
        with open(os.path.join(os.path.dirname(__file__), fname)) as _in:
            return _in.read()

    setup(
        name="Arignote",
        version="0.1",
        author="Stephen Hoover",
        author_email="Stephen.LD.Hoover hosted-on gmail.com",
        url="",
        description="Neural network models",
        packages=find_packages(),
        long_description=read('README.md')
    )

if __name__ == "__main__":
    main()
