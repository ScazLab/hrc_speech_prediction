# Action Prediction from Speech [![Build Status](https://travis-ci.org/ScazLab/hrc_speech_prediction.svg?branch=master)](https://travis-ci.org/ScazLab/hrc_speech_prediction) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/28d154d7ad7045aaa5a21eb9e6515f2c)](https://www.codacy.com/app/Baxter-collaboration/hrc_speech_prediction?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ScazLab/hrc_speech_prediction&amp;utm_campaign=Badge_Grade)

## Repository organization

### Scripts

- `archive/`: script related to older experiments (eventually move into directories for each of them)
- `controller*`: controllers
    - `controller.py`: current controller using the combined model
- `data/*`: scripts specific to data manipulation and preprocessing (consider splitting into subdirectories for each data set)
- `evaluate*`: model off-line evaluation
- `train*`: model training


### Data

A directory to store manually pre-processed data.


### Main modules

- `bag_parsing`: helpers to extract data from `rosbags`
- `combined_mode`
- `data`: helpers specific to the structure of data of the training user experiment
- `defaults`: paths to default data and model directory. This is to avoid hard-coded paths and a command line argument should always be preferred. Ideally only use for resources that need to be committed to the repository.
- `features`: helpers to extract speech and context features
- `models`
