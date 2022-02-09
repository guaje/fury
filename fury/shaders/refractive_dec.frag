uniform float absorption;
uniform float IOR1;
uniform float IOR2;

float FRESNEL_POW = 5.;

float ETA = IOR1 / IOR2;

// see http://en.wikipedia.org/wiki/Refractive_index Reflectivity
float R0 = ((IOR1 - IOR2) * (IOR1 - IOR2)) / ((IOR1 + IOR2) * (IOR1 + IOR2));
