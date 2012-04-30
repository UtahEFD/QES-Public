uniform sampler2DRect Wind;
uniform float max_vel;

void main(void)
{
  vec2 texCoord = gl_TexCoord[0].xy;
  vec4 wind_dir = vec4(texture2DRect(Wind, texCoord)); 
  
  float vel = sqrt(wind_dir.x*wind_dir.x + wind_dir.y*wind_dir.y + wind_dir.z*wind_dir.z);
  float normVel = vel / max_vel;

  vec3 layerColor; 

  vec3 MAXVEL = vec3(0.5, 0.0, 0.0);
  vec3 RED = vec3(1.0, 0.0, 0.0);
  vec3 YELLOW = vec3(1.0, 1.0, 0.0);
  vec3 CYAN = vec3(0.0, 1.0, 1.0);
  vec3 BLUE = vec3(0.0, 0.0, 1.0);
  vec3 ZEROVEL = vec3(0.0, 0.0, 0.5);

  // Mapping the following
  //  Dark Blue <--> Blue <--> Cyan <--> Yellow <--> Red <--> Dark Red
  //      0          0.25      0.50       0.75       1.0        > 1.0 
  float t = 0.0;
  if (normVel <= 0.25)
    {
      t = normVel * 4.0;  // scale to 0-1
      layerColor = mix(ZEROVEL, BLUE, t);
    }    
  else if (normVel <= 0.50) 
    { 
      t = 4.0*normVel + -1.0;
      layerColor = mix(BLUE, CYAN, t);
    }
  else if (normVel <= 0.80) 
    { 
      t = 4.0*normVel + -2.0;
      layerColor = mix(CYAN, YELLOW, t);
    }
  else if (normVel <= 1.0)
    { 
      t = 4.0*normVel + -3.0;
      layerColor = mix(YELLOW, RED, t);
    }
  else
    {
      // If the velocity is greater than our computed maximum
      // velocity, then give it a color that stands out... white for now.  The reason that a
      // vel may be larger than the max is because we computing the
      // max_vel to be the average + (3 or 4) standard deviations
      // away.  This lets us visualize abnormally large mean wind
      // velocities.
      t = 4.0*normVel + -4.0;
      layerColor = mix(RED, MAXVEL, t);
    }


  gl_FragColor = vec4(layerColor, .5);
}
