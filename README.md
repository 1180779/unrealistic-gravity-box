# Unrealistic gravity box

A simple cuda openGL interop project simulating an unrealistic gravity box. 

## Why unrealistic?
Energy is never lost in the simulation. All collisions are treated as perfect elastic collisions. 

Energy loss is easy to add and can be done by few manipulations in the code base. 

## Settings

Project supports a few user provided settings:
- number of particles
- maximum starting veclocity in x and y axis (in some arbitrary units - check code for details)
- acceleration `g`
- particles radius

Some of the notable limitations include: 
- all particles have the same radius

The settings can be loaded from file located in the same directory as the executable. 
Example [configuration](./Cuda_OpenGL_GLFW_glad_test/config.txt) is provided. 

## Examples

- 15 big particles
![](./samples/sample-15.mp4)

- 100 000 small particles (chaos)
![](./samples/sample-100000.mp4)

## FAQ

**Q: Why is the project folder named "Cuda_OpenGL_GLFW_glad_test"?**

**A:** It was a working name in early testing. Due to Visual Studio having issues with folder renaming I have not changed it. 

**Q: It seems like the particles are accelerating?**

**A:** It does seem like this is the case. I have no idea why thought. Feel free to investigate if you feel like it. 
