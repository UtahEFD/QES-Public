uniform sampler2DRect texSampler;

void main(void)
{
  vec2 texCoord = gl_TexCoord[0].xy;
  vec4 celltype = vec4(texture2DRect(texSampler, texCoord)); 
  
  //   float cP[11];
  //   cP[0] = 0.0;  // building
  //   cP[1] = 0.1;  // air
  //   cP[2] = 0.2;  // front
  //   cP[3] = 0.3;  // roof
  //   cP[4] = 0.4;  // cavity
  //   cP[5] = 0.5;  // wake
  //   cP[6] = 0.6;  // canyon
  //     cP[7] = 0.7;  // exp wake
//	   cP[8] = 0.8;  // veg
	   //  cP[9] = 0.9;  // blended
	   //  cP[10] = 1.0; // garage

  float t = celltype.x * 0.10;

  // Mapping the following
  //  Dark Blue <--> Blue <--> Cyan <--> Yellow <--> Red <--> Dark Red
  //      0          0.25      0.50       0.75       1.0        > 1.0 
  vec3 MAXVEL = vec3(0.5, 0.0, 0.0);
  vec3 RED = vec3(1.0, 0.0, 0.0);
  vec3 YELLOW = vec3(1.0, 1.0, 0.0);
  vec3 CYAN = vec3(0.0, 1.0, 1.0);
  vec3 BLUE = vec3(0.0, 0.0, 1.0);
  vec3 ZEROVEL = vec3(0.0, 0.0, 0.5);

  vec3 layerColor;
  if (t <= 0.25)
    {
      t = t * 4.0;  // scale to 0-1
      layerColor = mix(ZEROVEL, BLUE, t);
    }    
  else if (t <= 0.50) 
    { 
      t = 4.0*t + -1.0;
      layerColor = mix(BLUE, CYAN, t);
    }
  else if (t <= 0.80) 
    { 
      t = 4.0*t + -2.0;
      layerColor = mix(CYAN, YELLOW, t);
    }
  else if (t < 1.0)
    { 
      t = 4.0*t + -3.0;
      layerColor = mix(YELLOW, RED, t);
    }
  else
    {
      layerColor = MAXVEL;
    }

  gl_FragColor = vec4(layerColor, 1.0);
}
