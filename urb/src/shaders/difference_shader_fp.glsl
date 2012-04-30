uniform sampler2D difference_tex;
uniform float difference_threshold;
uniform float difference_alpha;

varying vec4 pcolor;

void main(void)
{
  vec2 texCoord = gl_TexCoord[0].st;

  float value = texture2D(difference_tex, gl_TexCoord[0].st);

   vec4 color = vec4(0.0,0.0,0.0,0.0);

	//Better hope it's low. You want to see nothing...
	if(value > difference_threshold) 
	{
		color.r = value;
		color.a = difference_alpha;
	}

	color.g += 0.05;
	color.a += difference_alpha / 5.;


  gl_FragColor = color;
}
