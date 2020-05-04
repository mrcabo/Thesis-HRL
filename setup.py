import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="thesishrl",
    version="0.0.1",
    packages=setuptools.find_packages(),
    # install_requires=['gym', 'householdenv', 'torch', 'matplotlib'],
    install_requires=['gym', 'torch', 'matplotlib'],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.json"]
    },
    # metadata to display on PyPI
    author="Diego Cabo Golvano",
    author_email="dcgdiego@gmail.com",
    description="This is my thesis work on HRL",  # TODO
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="Reinforcement Learning environment",
    url="https://github.com/mrcabo/Thesis-HRL.git",  # project home page, if any
    project_urls={
        "Source Code": "https://github.com/mrcabo/Thesis-HRL.git",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
