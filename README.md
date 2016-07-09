# touchFluid
###Performing dynamic systems

touchFluid explores computational fluid dynamics and reaction-diffusion using CUDA in TouchDesigner. It features Semi-Lagrangian fluid advection along a 2d grid with pressure, vorticity confinement, temperature, and obstacles. Density is coupled with a reaction-diffusion system which feeds back into the buoyancy of the velocity field to create unusual morphogenic patterns.

touchFluid is a key component in my Masters [thesis project](http://timesequence.blogspot.com/) for the Media Arts & Technology program at the University of California Santa Barbara.

#### Licensing
touchFluid code is released under the [MIT License](https://github.com/kamindustries/touchFluid/blob/master/LICENSE). 

Note: Two files required to build the CUDA .dll are restricted by Derivative Inc. to be shared with only authorized licensees of TouchDesigner, so they are not included in this repo. Send me a message if you want them and let Derivative know you would like them to ease up on their usage license for these files.
