"""Default locations to look for data and store models."""


import os


PACKAGE_ROOT = os.path.join(os.path.dirname(__file__), "..")
#PACKAGE_ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(PACKAGE_ROOT, "data")
MODEL_PATH = os.path.join(PACKAGE_ROOT, "models")
