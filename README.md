# CROCODILE

This package allows you to run atmospheric retrievals of gas giant exoplanets observed directly using cross-correlation spectroscopy along photometry or the usual spectroscopy. The methods are described in detail in Hayoz et al. 2023.

## Usage instructions

### Installation

This package makes use of petitRADTRANS. Therefore one needs to install petitRADTRANS first: see installation steps at https://petitradtrans.readthedocs.io/en/latest/content/installation.html
Moreover, one needs to install pymultinest: https://johannesbuchner.github.io/PyMultiNest/

After this is done, CROCODILE can be installed using by cloning the repository to your local machine:
```
git clone https://github.com:JHayoz/CROCODILE
```
Move to the directory and install the code using 
```
pip install .
```
Finally, depending on which opacity database is being used and where it is installed locally, one needs to tell petitRADTRANS where to look for the database and what the names of the opacities are. This is done by modifying the os environment throughout the code of CROCODILE
```
os.environ["pRT_input_data_path"] = "/your_absolute_path_to/petitRADTRANS/input_data"
```
To tell CROCODILE what the names of the opacities are, one needs to modify the file config_petitRADTRANS to contain the correct names. For example, with the standard installation of petitRADTRANS, the opacity of water is denoted by the name "H2O_main_iso", however if one is using another database where water is called "H2O_Chubb", then one needs to write that into the poor_mans_abunds_lbl and poor_mans_abunds_ck functions.


### Citations
This package was developed by Jean Hayoz from ETH Zürich. Please cite Hayoz et al. 2023 if you are using CROCODILE for your scientific analyses.