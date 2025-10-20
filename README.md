# Particle Filter Localization
CompRobo 2025 - Connor Hoang, Franklin Noble

### Overview
The goal of this project is to create a localization method for the neato robot, provided a map of the space that is accurate using only information from a lidar and odometry.

To do this, we implemented a particle filter in C++ using ROS2 middleware to effectively discretize a probability field and ultimately converge on the robots position.

Our final 
### Overall Code Structure
<!--(design decisions)-->
<!--Which files, explain c++...-->
Our primary particle filter logic is held in pf.cpp and the corresponding header file pf.hpp. We also have angle_helpers, helper_functions, and occupancy_field as files with functionality we use in pf.cpp related to their name.

<!-- insert here figure of pubs and subs -->

Within pf.cpp, we had a class called ParticleFilter which contained the fundamental operations for the particle filter implementation. 

### Particle Filter Logic 
<!--appraoch-->
<!--Explain conceptual logic behind how a particle filter works-->
Shown in Figure _ is the particle filter logic.

<!-- block diagram of pf logic--> 

#### Initialize Particles
Finds the bounding box of the map and generates random particles within those bounds in valid locations until _(the desired number of particles is reached)__.
#### Update Particles with Odom
This function finds the change in the robot's position since it was last updated. It then adds these changes in position to each particle as if each particle was the robot's position and orientation.  
#### Update Particles with laser (LIDAR)
The function parses the lidar data to determine the closest distance to an object. __mention threashould and infinite?__ Then the function determines for each particle in the particle cloud checks the closest distance to an obstacle at the same angle of the true lidar data. 
#### Normalize Particles
This function is straightforward -- it divides the weight of each particle by the total weight of all particles, so that all the weights add up to one.
#### Resample Particles

#### Calculate Pose

#### 

<!-- image of steps taken in run loop -->

### Challenges and Takeaways
One notable challenge has been adapting to C++, particuarly in a ROS enviorment. We both had some previous experience with C++ in firmware or otherwise, but we came into this project unfamiliar with some of the more complex elements of the language, such as pointers and the abstraction techniques that ROS uses. Most of the learnings here were from syntax and fundamental knowledge of how C++ works, especially as opposed to Python.  

Another challenge we faced was working with outside code we did not write. While relying on this code was incrediably useful, due to time contraints we were unable to fully understand every aspect of the provided code, and hence remained unconfident in modifying code, even when we encountered seemingly undesired behavior. As part of a workaround for this, we either made assumptions and verified them using vizualization or wrote code in our section that accounted for incorrect assumptions during runtime. 




### Next Steps
One of the biggest limitations we found was that the provided implementation of an occupancy field did not provide the angle to the nearest obstacle, meaning that our particles struggled to converge on the robot's orientation. Future work could explore a more robust angle calculation that makes fewer assumptions, either through an improved occupancy field implementation or using new methods. 
Although our choice to use C++ provided us with runtime optimization, our implementation leaves a lot of room for speed improvements. Right now, the filtering steps run just fast enough to localize the robot in real time -- this is good enough for small maps like the one we used, but localization on larger maps would likely need more particles to avoid dangers like particle death. Speed optimizations would let us use more particles, more resampling, and potentially even additional methods of evaluating particle likelihoods.

### Attribution of Work

### Additional Documentation
<!-- say where bag files are attached here-->
