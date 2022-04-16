## Federated Learning + Behavior Data + Authentication
A project that authenticates people based on their behavior data

## Install Requirements:
```pip3 install -r requirements.txt```

## Prepare Dataset:

## Run Experiments: 

The main file that does not have federated learning is "main-cp.py". The main file with federated learning is "main-fed.ipynb".

<!-- DATASET -->
<h2 id="dataset"> :floppy_disk: Dataset</h2>
<p> 
  BrainRun dataset. It can be found here:  https://zenodo.org/record/2598135#.X4H0qXgzbeo
  
   <!-- PREPROCESSING -->
<h2 id="preprocessing"> :hammer: Preprocessing</h2>

Data from motion sensors is in the format: x, y, z, screen. Where x, y, z is the position of the mobile device according to the 3 axes and the screen is the game that was recorded.
The dataset for each user consists of JSON files. Each JSON file is also a timestamp, during which data was collected.

In this way, the data collected by the sensors are related to a temporal dimension, so a windowing technique is used to segment the raw time series and extract resources from each segment.

<!-- PRE-PROCESSED DATA -->
<h2 id="preprocessed-data"> :diamond_shape_with_a_dot_inside: Pre-processed data</h2>

The dataset was segmented using a window size of considering 200 non-overlapping data points.
