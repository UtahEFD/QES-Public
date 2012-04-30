uniform sampler2D unitintervaldifference_tex;
uniform float unitinterval_alpha;

varying vec4 pcolor;

void main(void)
{
  vec2 texCoord = gl_TexCoord[0].st;

  float value = texture2D(unitintervaldifference_tex, gl_TexCoord[0].st);
  
   vec4 color = vec4(0.0,0.0,0.0,0.0);

/*
	// Do some nice shading interval stuff.
	How many colors? ...
*/

	//Fill in
	if(value > 0.51) 
	{
		color.g = value * 1.5 - .5;
		color.a = (value * 2.0 - 1.0) * 0.2;
		color.a = unitinterval_alpha;
	}
	else if(value < 0.49) 
	{
		color.b = 1.0 - value * 1.5;
		color.a = (1.0 - value * 2.0) * 0.2;
		color.a = unitinterval_alpha;
	}
	else 
	{
		color.r = 0.0;
		color.g = 0.0;
		color.b = 0.0;
		color.a = 0.0;
	}

	color.r += 0.05;
	color.a += 0.05;

  gl_FragColor = color;
}
