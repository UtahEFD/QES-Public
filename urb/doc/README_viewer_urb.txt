##### QUICurb Viewer Readme #############################

Author: Andrew Larson <lars2865@d.umn.edu>
Update: 3-3-2008

viewer_urb is intended to provide a way to see the 
lagrange data for QUICurb solver iteration by iteration. 
Within this readme are the keybindings that allow one to 
interactive with the environment representing said data.

The following binds are found in /Urb/viewer_urb.cpp.

<<<<< Key Bindings >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Esc	- Exit

==Movement==
	a - Strafe left
	d - Strafe right
	f - Strafe down
	r - Strafe up

	s - Move backwards
	w - Move forward

==Look==
	q - Turn left
	e - Turn right
	. - Look down
	, - Look up

==Data Manipulation==
	F - Toggle 2D / 3D view
	R - Reset the datamodule to iteration 0 (or 1?)
	D - Dump current iteration
	
	E - Toggle euler/lagragian view
	
	i - Do one iteration
	I - Iterate automatically to convergence

	k - Step through slices, starting at 0
	l - Step through slices backwards
	K - Show all slices

	( - Shift slices in -z direction
	) - Shift slices in +z direction
	{ - Decrease seperation of slices
	} - Increase seperation of slices

---------------------------------------------------------

#########################################################


