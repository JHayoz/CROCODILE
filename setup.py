import setuptools

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
    install_requires=['numpy','matplotlib','astropy','spectres','corner','jupyter','notebook','scipy','tqdm','progressbar','pandas','numba','seaborn','mendeleev','sklearn','mpi4py','PyAstronomy'],
    classifiers=['Programming Language :: Python :: 3'],
)