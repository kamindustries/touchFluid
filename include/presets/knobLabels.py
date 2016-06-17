dens diff	dens mult	N vis	audio mult
temp diff	dens mod1	lifespan	audio gain
vel diff	dens mod2	size	audio mod
delta time	dens mod3	softness	audio rand
	temp mult	color mix	shapes mult
N Jacobi	temp mod1	vel color	shapes edges
N Diffuse	temp mod2	spawn mode	copy offset
N React	temp mod3	N total	shapes rand
weight	vel mult	mass min	image mult
buoyancy	vel mod1	mass max	image edges
curl	vel mod2	mm min	image mod
	vel mod3	mm max	image rand
	pressure	hue	words mult
F	divergence	saturation	words edges
k	rand	value	words mod
RD eq	shader select	trails len	words rand
