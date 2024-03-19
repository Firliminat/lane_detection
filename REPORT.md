# Seoul Robotics Coding Assignment - Ego Lane Detection - Report - Alexandre BOBET

## Introduction

This is my report on how I tried to complete the lane detection assignment given by Seoul Robotics. At this point I have only produced data filtering code to visualize and understand more clearly the data sets of the assignement.

You can read INSTRUCTIONS.md to get a better understanding of the assignment.

## Familiarizing with the data

First I needed a better understanding of the data sets. Hence, I modified the data_visualization.py script to be able to filter the points based on their intensity or their lidar beam value on the fly. 

### Filtering on the intensity

I am looking for white lanes which are likely to be more reflective than other materials on the road. Therefore, ignoring points with low intensity should help.

Here are examples of visualizations with intensity filtering:

Frame 0 - intensity >= 15 :
![Low intensity filtering - frame 0 - min intensity 15](report_media\images\intensity_filtering_f0_int15.png)

Frame 5 - intensity >= 20 :
![Low intensity filtering - frame 5 - min intensity 20](report_media\images\intensity_filtering_f5_int20.png)

These visualizations confirms that points with higher intensity are more likely to belong to a lane. I also realize that they are false positives. For example, on the first image we can see a circle of highly reflective close to the center. It may be the back of the car. On the second image we can see a great number of false positives outside of the road. Maybe some surrounding buildings were highly reflective. Thus, filtering on intensity alone might not be enough.

### Filtering on the lidar beam value

I wanted to understand more clearly how are the lidar beams arranged and how the lidar beam value was related to the data. So I filtered the points for specific lidar beam values.

Here are examples of visualizations with lidar beam filtering:

Frame 0 - beam value = 62 :
![Beam value filtering - frame 0 - beam value 62](report_media\images\beamvalue_filtering_f0_beam62.png)

Frame 0 - beam value = 56 :
![Beam value filtering - frame 0 - beam value 56](report_media\images\beamvalue_filtering_f0_beam56.png)

It's at this point that I fully understood the schema explaining the data in the instructions. Beams are angled toward the ground each with a diferent angle. Thus, they give concentric observations of the environment. We can see on the second visualization that the circle of points is flatter on the top or bottom. Something higher than the floor must have blocked the view. As white lanes are likely to be close to the floor, filtering points closer or farther from the origin could help removing some false positives.

This kind of filtering could be really usefull for situations similar to the frame 5 where objects outside the road a reflecting lidar beams. If we isolate lidar beam points that hit objects outside the road we can see a really important variation in the distance to the center.

Frame 5 - beam value = 44 :
![Beam value filtering - frame 5 - beam value 44](report_media\images\beamvalue_filtering_f5_beam44.png)


## First idea

Starting from a set of points with (x,y,z) coordinates, reflection intensities and index of the beam which produced the point, the algorithm should find 3rd degree polynomial equations representing the closest white lanes of the road the lidar is on. When visualising the lane we realize that we can see the lanes in the spacial structure of the points with a higher intensity.

Therefore, my first intuition is that, with a good filtering of the point to keep only the ones most likely to belong to a white lane, it would be possible to use a clustering algorithm to separate the different lanes. Then we could fit a polynomial regression to each cluster.
