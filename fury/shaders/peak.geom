//VTK::System::Dec
/*
When activating a extension, there are two options: enable or require.
Require checks if the extension is available, and the enables it. Otherwise, it
will throw an unsupported extension error.
*/
#extension GL_ARB_arrays_of_arrays: require

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
in vec4 peaksVertexMCVSOutput[][$num_peaks];

// Convert points to line strips
layout(points) in;
layout(line_strip, max_vertices = $max_pnts) out;

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

    vertexColorGSOutput = vertexColorVSOutput[0];

    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    gl_Position = gl_in[0].gl_Position + 1;
    EmitVertex();
    EndPrimitive();

    int numPeaks = $num_peaks;

    /*
    for(int i = 0; i < numPeaks; i++)
    {
        if(currentVal > 0.0)
        {
            vec3 currentDir = peaksVertexMCVSOutput[0][i].xyz;
            //vec3 iniPoint = currentDir * currentVal + centerVertexMCVSOutput[0];
            vec3 iniPoint = currentDir * currentVal + gl_in[0].gl_Position.xyz;
            //vec3 endPoint = -currentDir * currentVal + centerVertexMCVSOutput[0];
            vec3 endPoint = -currentDir * currentVal + gl_in[0].gl_Position.xyz;
            vec3 diff = endPoint - iniPoint;
            //vertexColorGSOutput = vec4(orient2rgb(diff), 1);

            //gl_Position = vec4(iniPoint, gl_in[0].gl_Position.w) * MCDCMatrix;
            gl_Position = vec4(iniPoint, gl_in[0].gl_Position.w);
            EmitVertex();
            //gl_Position = vec4(endPoint, gl_in[0].gl_Position.w) * MCDCMatrix;
            gl_Position = vec4(endPoint, gl_in[0].gl_Position.w);
            EmitVertex();
            EndPrimitive();
        }
    }
    */
}
