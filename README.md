# Particle Filter Localization
CompRobo 2025 - Connor Hoang, Franklin Noble

### Overview
The goal of this project is to create a localization method for the neato robot, provided a map of the space that is accurate using only information from a lidar and odometry.

To do this, we implemented a particle filter with a Markov chain assumption to effectively discretize a probability field and ultimately converge on the robots position.

Our final 
### Overall Code Structure
<!--(design decisions)-->
<!--Which files, explain c++...-->
Our primary particle filter logic is held in pf.cpp and the corresponding header file pf.hpp. We also have angle_helpers, helper_functions, and occupancy_field as files with functionality we use in pf.cpp related to their name.

<!-- insert here figure of pubs and subs -->

Within pf.cpp, we had a class called ParticleFilter which contained the fundamental operations for the particle filter implementation. 

<!--  -->

### Particle Filter Logic 
<!--appraoch-->
<!--Explain conceptual logic behind how a particle filter works-->
Stepping thorugh our main loop, ___.

<!-- image of steps taken in run loop -->

### Challenges and Takeaways
One notable challenge has been adapting to C++, particuarly in a ROS enviorment. We both had some previous experience with C++ in firmware or otherwise, but we came into this project unfamiliar with some of the more complex elements of the language, such as pointers and the abstraction techniques that ROS uses. Most of the learnings here were from syntax and fundamental knowledge of how C++ works, especially as opposed to Python.  

Another challenge we faced was working with outside code we did not write. While relying on this code was incrediably useful, due to time contraints we were unable to fully understand every aspect of the provided code, and hence remained unconfident in modifying code, even when we encountered seemingly undesired behavior. As part of a workaround for this, we either made assumptions and verified them using vizualization or wrote code in our section that accounted for incorrect assumptions during runtime. 




### Next Steps
Future work could explore a more robust angle calculation that makes fewer assumptions, ___.
### Attribution of Work

### Additional Documentation
<!-- say where bag files are attached here-->
