varying vec4 pcolor;
varying vec2 cPosition;

void main(void)
{
  //vec4 pos = vec4(gl_Vertex);
  //gl_Position = gl_ModelViewProjectionMatrix * pos;
  gl_TexCoord[0] = gl_MultiTexCoord0;
  gl_Position = ftransform();

	
	
}
