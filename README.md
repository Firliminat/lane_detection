# Seoul Robotics Coding Assignment - Ego Lane Detection - Readme

## The project

This project is an assignment for Seoul Robotics in which the interviewee is asked to detect white lanes on a road using lidar values. At this point only visualization filters have been implemented.

### Line Detection - C++

First you'll need to have Eigen library. Make sure it is in your include path. Then you can compile :

```
g++ -g .\src\cpp\*.cpp -o .\bin\main.exe
```

Finally you can execute the program.

```
.\bin\main.exe
```

This program will load the data and write it's outputs from these relatives paths : .\pointclouds, .\sample_output.
You can edit them by changing the values of data_folder and lanes_folder if needed.

Once the data loaded the algortihm will loop through all the frames to find the lanes for each frame and generate the corresponding lane file.

You can use another model that will try to find more than two lanes by passing `--nPoly` as an argument when calling the program. This program will write the lane files to .\sample_output_nPoly.

### Visualization - Python

First you'll need to set up a python3.8 virtual environment. You can do it this way :

```
py -3.8 -m venv "lane_test"

# On Windows
.\lane_test\Scripts\activate

# On Unix or MacOS
source tutorial-env/bin/activate

pip install -r requirement.txt
```

#### Start it

To visualize the data and filter it you need to run data_visualize.py. You can do it this way:
```
python data_visualize.py
```

#### Control the visualization

- Change frame by using the left or right arrow keys.
- Increase or decrease intensity threshold with PgUp or PgDown keys. A threshold of 0 will display all points.
- Increase or decrease intensity ceiling with D or C keys. A ceiling of 255 will display all points.
- Increase or decrease the selected lidar beam value with up or down arow keys. A value of -1.0 will display all beams.