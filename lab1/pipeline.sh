#!/bin/bash

# pip install numpy pandas scikit-learn joblib

python data_creation.py
python data_preprocessing.py
python model_preparation.py
python model_testing.py