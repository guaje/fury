//fragOutput0 = vec4(irradiance, opacity);

//fragOutput0 = vec4(worldReflect, opacity);

vec3 worldRefract = normalize(envMatrix * refract(-V, N, ETA));
//fragOutput0 = vec4(worldRefract, opacity);

//float fresnel = R0 + (1. - R0) * pow(1. - dot(V, N), FRESNEL_POW);
float fresnel = R0 + (1. - R0) * pow((1. - NdV), FRESNEL_POW);
//float fresnel = R0 + (1. - R0) * pow(1. - 1., FRESNEL_POW);

//vec3 prefilteredColor = textureLod(prefilterTex, worldRefract, roughness * prefilterMaxLevel).rgb;
//fragOutput0 = vec4(prefilteredColor, opacity);

vec3 prefilteredColor = textureLod(prefilterTex,
    mix(worldRefract, worldReflect, fresnel),
    roughness * prefilterMaxLevel).rgb;
//fragOutput0 = vec4(prefilteredColor, opacity);

iblSpecular = prefilteredColor * specularBrdf;
//fragOutput0 = vec4(iblSpecular, opacity);

color = iblDiffuse + iblSpecular;
//fragOutput0 = vec4(color, opacity);

color += Lo;
//fragOutput0 = vec4(color, opacity);

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

/*
float df = max(0, normalVCVSOutput.z);
float sf = pow(df, specularPower);

diffuse = df * diffuseColor * lightColor0;
//fragOutput0 = vec4(diffuse, opacity);

specular = sf * specularColor * lightColor0;
//fragOutput0 = vec4(specular, opacity);

color = ambient + diffuse + specular;
//fragOutput0 = vec4(color, opacity);
fragOutput0 = vec4(color + prefilteredColor, opacity);

/*
float sigma = 30;
float thickness = 2;
float intensity = exp(-sigma * thickness);
//fragOutput0 = vec4(intensity * color, opacity);
*/
