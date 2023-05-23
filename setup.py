import setuptools

#Let us store all the info inside the readme file in the
#long description variable. This description will be shown in test.pypi webiste
with open("README.md",'r') as fh:
    long_description=fh.read()

setuptools.setup(
    name='CROCODILE',
    version='0.1',
    author='Jean Hayoz',
    author_email='jeanhayoz94@gmail.com',
    description='Run atmospheric retrievals of directly-imaged gas giant exoplanets using cross-correlation spectroscopy.',
    packages=setuptools.find_packages(),
    long_description = long_description,
    long_description_content_type='text/markdown',
    url='git@github.com:JHayoz/CROCODILE',
    install_requires=['numpy','matplotlib'],
    classifiers=['Programming Language :: Python :: 3'],
)