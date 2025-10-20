# Particle Filter Localization
CompRobo 2025 - Connor Hoang, Franklin Noble

### Overview
The goal of this project is to create a localization method for the neato robot, provided a map of the space is given.

To do this, we implemented a particle filter with a Markov chain assumption to effectively discretize a probability field and ultimately converge on the robots position.
### Overall Code Structure
<!--(design decisions)-->
<!--Which files, explain c++...-->
Our primary particle filter logic is held in pf.cpp and the corresponding header file pf.hp. We also have angle_helpers, helper_functions, and occupancy_field as files with functionality we use in __. 
### Particle Filter Logic 
<!--appraoch-->
<!--Explain conceptual logic behind how a particle filter works-->
When specifically requested by the user, the robot should listen to velocity commands passed by keypresses on the terminal running the code. This helps drive the robot to a desired location when testing autonomous mode, exiting the `bump` state if the bump sensor is triggered, or for any other reason in which manual control of the robot would be useful.

### Challenges and Takeaways
One notable challenge has been adapting to C++, particuarly in a ROS enviorment. We both had some previous experience with C++ in firmware or otherwise, but we came into this project unfamiliar with some of the more complex elements of the language, such as pointers and the abstraction techniques that ROS uses. Most of the learnings here were from syntax and fundamental knowledge of how C++ works, especially as opposed to Python.  

Another challenge we faced was working with outside code we did not write. While relying on this code was incrediably useful, due to time contraints we were unable to fully understand every aspect of the provided code, and hence remained unconfident in modifying code, even when we encountered seemingly undesired behavior. As part of a workaround for this, we either made assumptions and verified them using vizualization or wrote code in our section that accounted for incorrect assumptions during runtime. 




### Next Steps
Future work could explore a more robust angle calculation that makes fewer assumptions, ___.
### Attribution of Work

### Additional Documentation
