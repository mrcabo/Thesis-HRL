import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Thesis-HRL",
    version="0.0.1",
    install_requires=['gym', 'numpy', 'torch', 'matplotlib'],
    include_package_data=True,

    # metadata to display on PyPI
    author="Diego Cabo Golvano",
    author_email="dcgdiego@gmail.com",
    description="This is an environment for RL purposes",  # TODO
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="Reinforcement Learning environment",
    url="https://github.com/mrcabo/Household-env.git",  # project home page, if any
    project_urls={
        "Source Code": "https://github.com/mrcabo/Household-env.git",
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
