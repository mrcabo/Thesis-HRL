from setuptools import setup, find_packages

setup(
    name="Thesis-HRL",
    version="0.1",
    packages=find_packages(),
    scripts=["say_hello.py"],

    install_requires=['gym', 'numpy'],
    include_package_data=True,

    # metadata to display on PyPI
    author="Diego Cabo Golvano",
    author_email="dcgdiego@gmail.com",
    description="This is an Example Package",  # TODO
    keywords="hello world example examples",
    url="https://github.com/mrcabo/Thesis-HRL",  # project home page, if any
    project_urls={
        "Bug Tracker": "https://bugs.example.com/HelloWorld/",
        "Documentation": "https://docs.example.com/HelloWorld/",
        "Source Code": "https://github.com/mrcabo/Thesis-HRL",
    },
    classifiers=[
        "License :: OSI Approved :: Python Software Foundation License"
    ]

    # could also include long_description, download_url, etc.
)
