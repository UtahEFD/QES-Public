uniform sampler2DRect euler_u_tex;
uniform float euler_alpha;

varying vec4 pcolor;

void main(void)
{
  vec2 texCoord = gl_TexCoord[0].st;

  vec4 texValue = texture2DRect(euler_u_tex, gl_TexCoord[0].st);

  vec4 color = vec4(0.,0.,0.,0.);

	// U  
	if(texValue.r > .51) 
	{
		color.r +=  texValue.r * 1.5 - .5;
		color.a += euler_alpha / 5.;
	}
	else if(texValue.r < .49) 
	{
		color.g +=  1. - texValue.r * 1.5;
		color.a += euler_alpha / 5.;
	}
	else
	{
		color.a += euler_alpha / 15.;
	}

	// V
	if(texValue.g > .51) 
	{
		color.b += texValue.g * 1.5 - .5;
		color.a += euler_alpha / 5.;
	}
	else if(texValue.g < .49) 
	{
		color.rg += 1. - texValue.g * 1.5;
		color.a  += euler_alpha / 5.;
	}
	else
	{
		color.a += euler_alpha / 15.;
	}
	
	// W
	if(texValue.b > .51) 
	{
		color.rgb += texValue.b;
		color.a   += euler_alpha / 5.;
	}
	if(texValue.b < .49) 
	{
		color.rgb -=  .5 - texValue.b;
		color.a   += euler_alpha / 5.;
	}
	else
	{
		color.a += euler_alpha / 15.;
	}
	
  gl_FragColor = color;
}
