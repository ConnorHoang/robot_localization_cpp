# Particle Filter Localization
CompRobo 2025 - Connor Hoang, Franklin Noble

## Overview
The goal of this project is to create a localization method for the neato robot, provided a map of the space is given.

To do this, we implemented a working particle filter using a Markov chain to effectively discretize a probability field and ultimately converge on the robots position.
### Overall Code Structure
<!--(design decisions)-->
<!--Which files, explain c++...-->
#### Particle Filter Logic 
<!--appraoch-->
<!--Explain conceptual logic behind how a particle filter works-->
When specifically requested by the user, the robot should listen to velocity commands passed by keypresses on the terminal running the code. This helps drive the robot to a desired location when testing autonomous mode, exiting the `bump` state if the bump sensor is triggered, or for any other reason in which manual control of the robot would be useful.

## Conclusion

### Takeaways
<!--(lessons learned)-->

### Challenges
One major challenge has been adapting to C++, particuarly in a ROS enviorment. We both have previous experience with C++ in firmware or otherwise, ____.

### Next Steps

### Attribution of Work

### Additional Documentation
