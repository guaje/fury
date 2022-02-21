//VTK::System::Dec
//VTK::PositionVC::Dec
uniform mat4 MCDCMatrix;

//VTK::PrimID::Dec

// declarations below aren't necessary because
// they are already injected by PrimID template
// this comment is just to justify the passthrough below
//in vec4 vertexColorVSOutput[];
//out vec4 vertexColorGSOutput;

//VTK::Color::Dec
//VTK::Normal::Dec
//VTK::Light::Dec
//VTK::TCoord::Dec
//VTK::Picking::Dec
//VTK::DepthPeeling::Dec
//VTK::Clip::Dec
//VTK::Output::Dec

in vec3 centerVertexMCVSOutput[];
in vec3 dirVertexMCVSOutput[];
in vec4 peaksVertexMCVSOutput[][];
in vec4 vertexMCVSOutput[];

// Convert points to line strips
layout(points) in;
layout(line_strip, max_vertices = 4) out;

vec3 orient2rgb(vec3 v)
{
    float r = sqrt(dot(v, v));
    if (r != 0)
        return abs(v / r);
    return vec3(1);
}

void main()
{
    /*
    //vec3 iniPoint = dirVertexMCVSOutput[0] * 1 + centerVertexMCVSOutput[0];
    vec3 iniPoint = dirVertexMCVSOutput[0] * 1 + gl_in[0].gl_Position.xyz;
    //vec3 endPoint = -dirVertexMCVSOutput[0] * 1 + centerVertexMCVSOutput[0];
    vec3 endPoint = -dirVertexMCVSOutput[0] * 1 + gl_in[0].gl_Position.xyz;
    vec3 diff = endPoint - iniPoint;
    vertexColorGSOutput = vec4(orient2rgb(diff), 1);

    //gl_Position = vec4(iniPoint, vertexMCVSOutput[0].w) * MCDCMatrix;
    gl_Position = vec4(iniPoint, gl_in[0].gl_Position.w);
    EmitVertex();
    //gl_Position = vec4(endPoint, vertexMCVSOutput[0].w) * MCDCMatrix;
    gl_Position = vec4(endPoint, gl_in[0].gl_Position.w);
    EmitVertex();
    EndPrimitive();
    */
    for(int i = 0; i < 2; i++)
    {
        vec3 iniPoint = dirsVertexMCVSOutput[i].xyz * 1 + gl_in[0].gl_Position.xyz;
        vec3 endPoint = -dirsVertexMCVSOutput[i].xyz * 1 + gl_in[0].gl_Position.xyz;
        vec3 diff = endPoint - iniPoint;
        vertexColorGSOutput = vec4(orient2rgb(diff), 1);

        gl_Position = vec4(iniPoint, gl_in[0].gl_Position.w);
        EmitVertex();
        gl_Position = vec4(endPoint, gl_in[0].gl_Position.w);
        EmitVertex();
        EndPrimitive();
    }
}
