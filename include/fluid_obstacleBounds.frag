// Precompute the obstacle neighborhood to use in Jacobi and gradient subtraction
// Idea taken from PixelFlow, Thomas Diewald (MIT License): https://github.com/diwi/PixelFlow

out vec4 fragColor;

int OBSTACLE = 0;

void main()
{
    ivec2 T = ivec2(gl_FragCoord.xy);

    fragColor = vec4(0.);

    // Find neighboring obstacles:
    fragColor.x = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(0, 1)).b;
    fragColor.y = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(0, -1)).b;
    fragColor.z = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(1, 0)).b;
    fragColor.w = texelFetchOffset(sTD2DInputs[OBSTACLE], T, 0, ivec2(-1, 0)).b;
}
