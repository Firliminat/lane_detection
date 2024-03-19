# Seoul Robotics Coding Assignment - Ego Lane Detection - Readme

## Detecting lanes using lidar data

This project is an assignment for Seoul Robotics in which the interviewee is asked to detect white lanes on a road using lidar values. At this point only visualization filters have been implemented.

## Setting Up

First you'll need to set up a python3.8 virtual environment. You can do it this way :

```
py -3.8 -m venv "lane_test"

# On Windows
.\lane_test\Scripts\activate

# On Unix or MacOS
source tutorial-env/bin/activate

pip install -r vis_requirement.txt
```

## Usage

### Starting the visualization

To visualize the data and filter it you need to run data_visualize.py. You can do it this way:
```
python data_visualize.py
```

### Controls

- You can chane frame by using the left or right arrow keys.
- You can increase or decrease intensity minimal value with PgUp or PgDown keys. A minimal value of 0 will display all points.
- You can increase or decrease lidar beam value with up or down arow keys. A value of -1.0 will display all beams.