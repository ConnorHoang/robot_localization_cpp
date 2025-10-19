# Particle Filter Localization
CompRobo 2025 - Connor Hoang, Franklin Noble

## Overview
The goal of this project is to create a localization method for the neato robot, provided a map of the space is given.

To do this, we implemented a working particle filter using a Markov chain to effectively discretize a probability field and ultimately converge on the robots position.
### Overall Code Structure
<!--Which files, explain c++...-->
#### Particle Filter Logic
<!--Explain conceptual logic behind how a particle filter works-->
When specifically requested by the user, the robot should listen to velocity commands passed by keypresses on the terminal running the code. This helps drive the robot to a desired location when testing autonomous mode, exiting the `bump` state if the bump sensor is triggered, or for any other reason in which manual control of the robot would be useful.
#### Implementation (Functions)
<!--Explain each function (do we need to explain what we didn't write?)-->
##### moved_far_enough_to_update
##### update_robot_pose
##### update_particles_with_odom
<!--for all functions-->

## Conclusion

### Takeaways

### Challenges
One major challenge has been adapting to C++, particuarly in a ROS enviorment. We both have previous experience with C++ in firmware or otherwise, ____.

### Next Steps

### Attribution of Work

### Additional Documentation

### Code Usage
