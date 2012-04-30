uniform sampler2DRect lagrange_tex;
uniform float lagrange_alpha;

varying vec4 pcolor;

void main(void)
{
  vec2 texCoord = gl_TexCoord[0].st;

  vec4 texValue = texture2DRect(lagrange_tex, gl_TexCoord[0].st);
  float value = texValue.b; 

   vec4 color = vec4(0.,0.,0.,0.);

	if(value > .51) 
	{
		color.g = value * 1.5 - .5;
		color.a = lagrange_alpha;
	}
	
	else if(value < .49) 
	{
		color.b = 1. - value * 1.5;
		color.a = lagrange_alpha;
	}
	
	else 
	{
		color.r = .05;
		color.gb = vec2(0.,0.);
		color.a = lagrange_alpha / 5.;
	}
	
  gl_FragColor = color;
}
