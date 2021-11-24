/*
// VC position of this fragment. This should not branch/return/discard.
vec4 vertexVC = vertexVCVSOutput;

// Place any calls that require uniform flow (e.g. dFdx) here.
vec3 fdx = vec3(dFdx(vertexVC.x),dFdx(vertexVC.y),dFdx(vertexVC.z));
vec3 fdy = vec3(dFdy(vertexVC.x),dFdy(vertexVC.y),dFdy(vertexVC.z));

// Generate the normal if we are not passed in one
fdx = normalize(fdx);
fdy = normalize(fdy);
vec3 normalVCVSOutput = normalize(cross(fdx,fdy));
if(cameraParallel == 1 && normalVCVSOutput.z < 0.0)
    normalVCVSOutput = -1.0*normalVCVSOutput;
if(cameraParallel == 0 && dot(normalVCVSOutput,vertexVC.xyz) > 0.0)
    normalVCVSOutput = -1.0*normalVCVSOutput;

const float prefilterMaxLevel = float(4);

vec3 albedo = pow(diffuseColor, vec3(2.2));

// TODO: roughness uniform

float ao = 1.0;

vec3 emissiveColor = vec3(0.0);

vec3 N = normalVCVSOutput;

vec3 V = normalize(-vertexVC.xyz);

float NdV = clamp(dot(N, V), 1e-5, 1.0);

vec3 irradiance = texture(irradianceTex, envMatrix*N).rgb;

vec3 worldReflect = normalize(envMatrix*reflect(-V, N));
*/
//fragOutput0 = vec4(worldReflect, opacity);

vec3 worldRefract = normalize(envMatrix * refract(-V, N, ETA));
//fragOutput0 = vec4(worldRefract, opacity);

//float fresnel = R0 + (1. - R0) * pow(1. - dot(V, N), FRESNEL_POW);
float fresnel = R0 + (1. - R0) * pow((1. - NdV), FRESNEL_POW);
//float fresnel = R0 + (1. - R0) * pow(1. - 1., FRESNEL_POW);

//vec3 prefilteredColorV3 = textureLod(prefilterTex, worldRefract, roughness * prefilterMaxLevel).rgb;
//fragOutput0 = vec4(prefilteredColorV3, opacity);
//fragOutput0 = vec4(mix(prefilteredColorV3, prefilteredColor, fresnel), opacity);
//fragOutput0 = vec4(prefilteredColor, opacity);
prefilteredColor = textureLod(prefilterTex, mix(worldRefract, worldReflect, fresnel), roughness * prefilterMaxLevel).rgb;
//fragOutput0 = vec4(prefilteredColor, opacity);
//fragOutput0 = vec4(prefilteredColorV3, opacity);

//fragOutput0 = vec4(specularColor, opacity);
//fragOutput0 = vec4(specularColorUniform, opacity);

//fragOutput0 = vec4(diffuseColorUniform, opacity);

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
