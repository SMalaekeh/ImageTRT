# This is the main module of the application
# I create synthetic data for downstream causal inference tasks

# main directory
import pathlib
DATA_DIR = pathlib.Path('/Users/sayedmorteza/Library/CloudStorage/Box-Box/Hetwet_Data')         

# folders
folders = {
    'wet' : DATA_DIR/'WETLAND_DEV_1996_2016',
    'dem' : DATA_DIR/'DEM',
    'cap' : DATA_DIR/'CAPITAL_1996',
    'claims_2016' : DATA_DIR/'LOG_CLAIMS_2016',
    'claims_1996' : DATA_DIR/'LOG_CLAIMS_2017',
}

