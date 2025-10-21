# Particle Filter Localization
CompRobo 2025 - Connor Hoang, Franklin Noble

### Overview
The goal of this project is to create a localization method for the neato robot, provided a map of the space that is accurate using only information from a lidar and odometry.

To do this, we implemented a particle filter in C++ using ROS2 middleware to effectively discretize a probability field and ultimately converge on the robots position.

Our final 
### Overall Code Structure
<!--(design decisions), system architecture-->
<!--Which files, explain c++...-->
Our primary particle filter logic is held in pf.cpp and the corresponding header file pf.hpp. We also have angle_helpers, helper_functions, and occupancy_field as files with functionality we use in pf.cpp related to their name.

Our code communicates with the robot and with visualization tools using ROS topics. The particle filter code itself is a ROS node, and it subscribes to the laser scan topic published by the robot. Whenever it updates the position estimate, it publishes that estimate to the transform `/tf` topic. It also publishes the particle cloud of pose guesses for debugging purposes. This topic, as well as the provided map, are displayed using the rviz2 ROS visualization tool. Rviz can also be used to publish an initial guess for the robot's position, so the convergence of the robot can be tested without needing large amounts of particles on larger maps. Separately, a teleoperation ROS node publishes move commands to the `/cmd_vel` topic, letting the robot drive around to gather more data about its environment. 
<!-- insert here figure of pubs and subs -->

Within pf.cpp, we had a class called ParticleFilter which contained the fundamental operations for the particle filter implementation. 

<!-- image of steps taken in run loop (the flow)-->

### Particle Filter Logic 
<!--appraoch-->
<!--Explain conceptual logic behind how a particle filter works-->
Shown in Figure _ is the particle filter logic.

<!-- block diagram of pf logic--> 

#### Initialize Particles
Finds the bounding box of the map. Next, if given a pose estimate generates particles randomly around that estimate. Otherwise, generates random particles within map bounds in valid locations until the desired number of particles is reached.
#### Update Particles with Odom
This function finds the change in the robot's position since it was last updated. It then adds these changes in position to each particle as if each particle was the robot's position and orientation.  
#### Update Particles with laser (LIDAR)
The function parses the LIDAR data to determine the closest distance and corresponding angle to an object. Then for each particle in the particle cloud, the function projects the true robot's closest distance out at the right angle. A helper function is called to get the closest distance from the projected point to an obstacle. As the ideal value would be in an object (and thereby zero), we proceed to weigh each particle such that the values closer to zero are more heavily weighted.
#### Normalize Particles
This divides the weight of each particle by the total weight of all particles, so that all the weights add up to one.
#### Resample Particles
Sort the particles in the particle cloud by weight and remove a set percentage of the lowest weighted particles (default 25%). Replace some percentage (default 20%) of the original particles as duplicates of the surviving particles with random noise, where surviving particles with higher weights have proportionally more particles assigned to them. Generate the remaining particles randomly as a means to mitigate dangers of particle death.
#### Calculate Pose
The robot's pose is estimated by calculating the weighted average of all particles in the particle cloud. The final x and y coordinates are the weighted mean of the particles' positions. The final orientation (theta) is calculated and weighted with angle wrapping in mind. The resulting pose represents the most likely position and orientation of the robot.
#### Check inbounds
If a particle in the particle cloud is outside the map bounds, generate a random particle to replace it.

<!-- image of steps taken in run loop -->

### Challenges and Takeaways
One notable challenge has been adapting to C++, particuarly in a ROS enviorment. We both had some previous experience with C++ in firmware or otherwise, but we came into this project unfamiliar with some of the more complex elements of the language, such as pointers and the abstraction techniques that ROS uses. Most of the learnings here were from syntax and fundamental knowledge of how C++ works, especially as opposed to Python.  

Another challenge we faced was working with outside code we did not write. While relying on this code was incrediably useful, due to time contraints we were unable to fully understand every aspect of the provided code, and hence remained unconfident in modifying code, even when we encountered seemingly undesired behavior. As part of a workaround for this, we either made assumptions and verified them using vizualization or wrote code in our section that accounted for incorrect assumptions during runtime. 




### Next Steps
In out current simulated setup, our particle filter converges on the correct location, but it does have accuracy and precision limitations. Future progress could explore tuning the different parameters available to us to find optimal values. It could also explore slightly different methods of weighting such as incorporating more directions of LIDAR data, or adding random noise to the odometry particle updates.

While our choice to use C++ provided us with runtime optimization, our implementation still leaves a lot of room for speed improvements. Right now, the filtering steps run just fast enough to localize the robot in real time -- this is good enough for small maps like the one we used, but localization on larger maps would likely need more particles to avoid dangers like particle death. Speed optimizations would let us use more particles, more resampling, and potentially even additional methods of evaluating particle likelihoods.

### Additional Documentation
<!-- say where bag files are attached here-->




















