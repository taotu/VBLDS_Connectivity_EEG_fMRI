# VBLDS_Connectivity_EEG_fMRI
This repo contains Matlab code for NeuRIPS 2019 paper. 

The data used by in the paper is available at https://www.dropbox.com/sh/15sltkkwb08w7xg/AADYY6PqzKeegxWu-AWbnfGXa?dl=0.

The data directory contains preprocessed simultaneous EEG-fMRI data from 10 subjects as described in the paper.

## Data structure
data.EEG: is the preprocessed EEG data used by the algorithm;

data.fMRI: upsampled BOLD timeseries;

data.m_category: modulatory inputs (face, car, house);

data.stimOnsetTime: stimulus onset timing;

data.L: precomputed lead-field matrix;

data.G: binary indicator matrix;
