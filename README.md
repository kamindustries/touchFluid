# touchFluid
### Fluids in TouchDesigner and GLSL

touchFluid is a lightweight 2d Semi-Lagrangian fluid solver for TouchDesigner. It features vorticity confinement, temperature, buoyancy, and obstacles.

The repo has a .toe project file and a .tox component file. The project file shows examples of impulses and obstacles: holding 1, 2, and 3 on the keyboard will add velocity, density, and temperature at the mouse position. The tox is just the core simulation component with inputs and outputs for the advection of velocity, density/temperature, and RGBA color fields.

I have added the shaders as separate files for easy editing and versioning, though they are not required to run either the .toe or .tox.

touchFluid is a key component in my Masters [thesis project](http://timesequence.blogspot.com/) for the Media Arts & Technology program at the University of California Santa Barbara.

This was originally written in CUDA. That repo was forked and can be found here: [touchFluidCUDA](https://github.com/kamindustries/touchFluidCUDA).

#### References
1. Jos Stam, [_Stable Fluids_](http://dl.acm.org/citation.cfm?id=311548).
2. Ronald Fedikw, Jos Stam, and Henrik Wann Jensen, [_Visual Simulation of Smoke_](http://dl.acm.org/citation.cfm?id=383260).
3. Mark Harris, [_Fast Fluid Dynamics Simulation on the GPU_](http://http.developer.nvidia.com/GPUGems/gpugems_ch38.html).
3. Philip Rideout, [_Simple Fluid Simulation_](http://prideout.net/blog/?p=58).

#### Licensing
touchFluid code is released under the [MIT License](https://github.com/kamindustries/touchFluid/blob/master/LICENSE).
