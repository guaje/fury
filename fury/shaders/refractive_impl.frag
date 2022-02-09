//fragOutput0 = vec4(irradiance, opacity);

//fragOutput0 = vec4(worldReflect, opacity);

vec3 worldRefract = normalize(envMatrix * refract(-V, N, ETA));
//fragOutput0 = vec4(worldRefract, opacity);

//float fresnel = R0 + (1. - R0) * pow(1. - dot(V, N), FRESNEL_POW);
float fresnel = R0 + (1. - R0) * pow((1. - NdV), FRESNEL_POW);
//float fresnel = R0 + (1. - R0) * pow(1. - 1., FRESNEL_POW);

/*
vec3 prefilteredColor = textureLod(prefilterTex, worldRefract,
    roughness * prefilterMaxLevel).rgb;
//fragOutput0 = vec4(prefilteredColor, opacity);
*/

vec3 prefilteredColor = textureLod(prefilterTex,
    mix(worldRefract, worldReflect, fresnel),
    roughness * prefilterMaxLevel).rgb;
//fragOutput0 = vec4(prefilteredColor, opacity);

//Shadertoy demo: https://www.shadertoy.com/view/MsByWG
// TODO: Add depth information for the absorption
//float z = gl_FragCoord.z / gl_FragCoord.w;
//fragOutput0 = vec4(vec3(z), opacity);
//float distanceToCamera = 1.0 / gl_FragCoord.w;
//fragOutput0 = vec4(vec3(distanceToCamera), opacity);

fragOutput0 = vec4(vec3(gl_FragCoord.z), opacity);

vec3 absorbedColor = (vec3(1) - albedo) * absorption * -gl_FragCoord.z;
//fragOutput0 = vec4(absorbedColor, opacity);

absorbedColor = exp(absorbedColor);
//fragOutput0 = vec4(absorbedColor, opacity);

color = prefilteredColor * absorbedColor;
//fragOutput0 = vec4(color, opacity);

/*
float depthScale = 1;

float depth = depthScale * gl_FragCoord.z;
fragOutput0 = vec4(depth, 0, 0, opacity);
*/

/*
float sigma = 30.0;
float intensity = exp(-sigma * thickness);
//fragOutput0 = vec4(intensity * albedo, opacity);
*/

/*
//Reuse VTKs PBR code
iblSpecular = prefilteredColor * specularBrdf;
//fragOutput0 = vec4(iblSpecular, opacity);

color = iblDiffuse + iblSpecular;
//fragOutput0 = vec4(color, opacity);
*/

/*
// VTK's Phong
float df = max(0, normalVCVSOutput.z);
float sf = pow(df, specularPower);

diffuse = df * diffuseColor * lightColor0;
//fragOutput0 = vec4(diffuse, opacity);

specular = sf * specularColor * lightColor0;
//fragOutput0 = vec4(specular, opacity);

color = ambient + diffuse + specular;
//fragOutput0 = vec4(color, opacity);
fragOutput0 = vec4(color + prefilteredColor, opacity);
*/

color = mix(color, color * ao, aoStrengthUniform);
//fragOutput0 = vec4(color, opacity);

color += emissiveColor;
//fragOutput0 = vec4(color, opacity);

// HDR tonemapping
//color = color / (color + vec3(1.));
//fragOutput0 = vec4(color, opacity);

// Gamma correction
color = pow(color, vec3(1. / 2.2));
fragOutput0 = vec4(color, opacity);
