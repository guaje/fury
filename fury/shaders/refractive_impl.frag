//float fresnel = R0 + (1. - R0) * pow(1. - dot(V, N), FRESNEL_POW);
float fresnel = R0 + (1. - R0) * pow((1. - NdV), FRESNEL_POW);
//float fresnel = R0 + (1. - R0) * pow(1. - 1., FRESNEL_POW);

vec3 worldRefract = normalize(envMatrix * refract(-V, N, ETA));
//fragOutput0 = vec4(worldReflect, opacity);
//fragOutput0 = vec4(worldRefract, opacity);

//vec3 prefilteredColorV3 = textureLod(prefilterTex, worldRefract, roughness * prefilterMaxLevel).rgb;
//fragOutput0 = vec4(prefilteredColorV3, opacity);
//fragOutput0 = vec4(mix(prefilteredColorV3, prefilteredColor, fresnel), opacity);
prefilteredColor = textureLod(prefilterTex, mix(worldRefract, worldReflect, fresnel), roughness * prefilterMaxLevel).rgb;
fragOutput0 = vec4(prefilteredColor, opacity);
//fragOutput0 = vec4(prefilteredColorV3, opacity);

//fragOutput0 = vec4(specularColor, opacity);
//fragOutput0 = vec4(specularColorUniform, opacity);

//fragOutput0 = vec4(diffuseColorUniform, opacity);
//fragOutput0 = vec4(specularColorUniform, opacity);

/*
float specF = pow(NdV, 10.);

//fragOutput0 = vec4(DiffuseLambert(albedo) * NdV, opacity);
vec3 LoTmp = vec3(DiffuseLambert(albedo) * NdV + vec3(specF));
//fragOutput0 = vec4(LoTmp, opacity);

vec3 ambientV3 = irradiance * DiffuseLambert(albedo) * NdV + prefilteredColorV3;
//fragOutput0 = vec4(ambient, opacity);
//fragOutput0 = vec4(ambientV3, opacity);

vec3 colorTmp = ambientV3;
colorTmp = mix(colorTmp, colorTmp * ao, aoStrengthUniform);
colorTmp += emissiveColor;
// HDR tonemapping
colorTmp = colorTmp / (colorTmp + vec3(1.));
// Gamma correction
colorTmp = pow(colorTmp, vec3(1. / 2.2));
fragOutput0 = vec4(colorTmp, opacity);
*/
