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


## First idea - N Polynomial Clustering

### The principle

Starting from a set of points with (x,y,z) coordinates, reflection intensities and index of the beam which produced the point, the algorithm should find 3rd degree polynomial equations representing the closest white lanes of the road the lidar is on. When visualising the lane we realize that we can see the lanes in the spacial structure of the points with a higher intensity.

Therefore, my first intuition is that, with a good filtering of the point to keep only the ones most likely to belong to a white lane, it would be possible to use a clustering algorithm to separate the different lanes. Then we could fit a polynomial regression to each cluster.

The idea is to perform a clustering algorithm similar to K-means but with polynomials instead of points as centers:

1. Initialize n polynomial regression
2. Attribute each point to the closest polynomial
3. Fit the regressions with it's new point cloud
4. Repeat 2 and 3 until the sum of the square distances to doesn't improove anymore
5. Repeat 1 with n+1 polynomials

### Selecting the points

#### Intensity

I filtered the point to remove low values of intensity as they are certainly not part of a lane. I used the intensity as a weight for the polynomial regression, favorising points with igh intensity. I tried different transformations on the intensity. Such as re-scaling it linearly to [0, 1], clipping the maximum to avoid points being too important or applying a logistic function to create a soft-step input. In the end keeping only the filtering was the best for this model.

#### Other filters

I tried building other filters based on the height or the distance to rhe origin but it did not help.

### Cluster number selection issue

With a manually fixed number of polynomials this algorithm can have really promising results.

Frame 0 - 6 lanes :
![Frame 0 - 6 lanes](report_media\images\nPolyf16l.png)

But finding this number automatically is not easy.

Frame 0 - 10 lanes :
![Frame 0 - 10 lanes](report_media\images\nPolyf110l.png)

#### Squared Residuals

We can't use the sum of squared residuals as a metric to stop adding clusters as it will always decrease when performing the iterations. Therefore, I first tried using the elbow method, but selecting a minimal value for score improvement that generalised well was not possible. So I tried to find the number of clusters that had the higest curvature by approximating with the second derivative of the squared residuals. But it didn't gave satisfying results.

#### Coefficient of determination

I decided to choose an interval for the number of polynomials then go through the whole interval and then choose the number of polnomials with the maximum of the minimums of the coefficient of determination of the lanes.

Here is a plot of the extremums and average of the R2 score for the frame 0: 
![Frame 0 - R2](report_media\images\nPolyf0r2.png)

It does not choose a good number of cluster. Here it would pick 8, the maximum it tried.

#### Sillhouette score

My next try was the silhouette score and it improved a bit the situation. 

Silhouette scores - Frame 0 :
![Frame 0 - Silhouette](report_media\images\nPolyf0silh.png)

As you can see this time it selects 3 lanes, the minimum tried.

### Other limitations

Reflective objects out of the road where a real issue when trying to fit the models. Some lanes are sticking to thiese objects thus getting away from the road.

Frame 10 :
![Frame 10 - Reflective object](report_media\images\nPolyf10.png)

### Implementation

I made a C++ implementation of this idea. You can find it in the MultiPolynomialRegression class. It uses the PolynomialRegression class for representing the lanes. The PolynomialRegression class is just a layer around the LinearRegression class creating the vector of polynomial features based on the inputs.

## 2nd idea - 2 lanes clustering

### The principle

The two main issus with the precedent ideas are the selection of the number of clusters ans the points outside the road trapping our models. As the task is to finc the two lanes closest to the origin, we could focus on those points and try to fit only two lines to them. This could fix our two issues.

The idea is to use polynomial regression on the whole point cloud and anchor it to the origin to get a profile for the road. And then filter the points based on their distances to this profile. Then we only need to fit two lanes.

### Selecting the width of the lane

When we filter points based on their distances to the profile we define a width and a center for the lane as the car could be changing lanes. Therefore, we need to select these parameters. As I did previously, I iterated other some lanes widths and center and kept the combination with the best silhouette score.

Silhouette VS Lane width :
![Frame 10 - Reflective object](report_media\images\silhouette_lanewisth.png)

In this graph we can see that the maximum silhouette seems to be a good score. But it does not generalises well. The algorithm tends to get the lanes too close and one is fitting to all the points when the other ones has really few.

## Conclusion

### Possible improvments

The two models I built do not generalize well to the frames. With the right parameters they cans fit really well to some simple frames.

I think that building better filters to identify more definitively wich points are part of the clusters would be really intersting.

Eliminating clusters of parasite points out of the road would surely improove the precision of the model.

Finding a good metric to assess the performances of the model would really help to tune the parameters of the models.
