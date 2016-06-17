	if ( mouse[2] < 1.0 && mouse[3] < 1.0 ) return;

	int i, j = dimX * dimY;
	i = (int)(mouse[0]*dimX-1);
	j = (int)(mouse[1]*dimY-1);

	float x_diff = mouse[0]-mouse_old[0];
	float y_diff = mouse[1]-mouse_old[1];
	//printf("%f, %f\n", x_diff, y_diff);

	if (i<1 || i>dimX || j<1 || j>dimY ) return;

	if (mouse[2] > 0.0 && mouse[3] > 0.0) {
		AddFromUI<<<grid,threads>>>(_u, x_diff * force, dt, i, j, dimX, dimY);
		AddFromUI<<<grid,threads>>>(_v, y_diff * force, dt, i, j, dimX, dimY);
	}

	if (mouse[3] > 0.0) {
		AddFromUI<<<grid,threads>>>(_dens, source_density, dt, i, j, dimX, dimY);
		AddFromUI<<<grid,threads>>>(_temp, source_temp, dt, i, j, dimX, dimY);
		//GetFromUI<<<grid,threads>>>(_chemB0, source_density, i, j, dimX, dimY);
		//particleSystem.addParticles(mouse[0], mouse[1], 100, .04);
	}

	if (mouse[4] > 0.0) printf("mouse[4] is down!\n");

	for (int i=0; i<6; i++){
		mouse_old[i]=mouse[i];
	}
