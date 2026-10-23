// optixu_stub.cpp - DG Bridge, stage 1: replacement for optixu.1.dll
//
// Design Garage's scenixrt8.dll looks up 6 functions from this library.
// The two simple helpers are implemented; the rest (runtime CUDA compilation,
// clustered meshes, the traversal API) report RT_ERROR_NOT_SUPPORTED and are
// logged loudly, so we find out if the demo ever relies on them.
// Logging goes through optix.1.dll (our stub) so everything ends up in one log.

#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>

#define API extern "C"
typedef unsigned int RTresult;
typedef size_t RTsize;

enum : unsigned { RT_SUCCESS = 0, RT_ERROR_INVALID_VALUE = 0x501, RT_ERROR_NOT_SUPPORTED = 0xA00 };

static void logLine(const char* fmt, ...)
{
    typedef void (*LogFn)(const char*);
    static LogFn fn = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
        HMODULE m = GetModuleHandleA("optix.1.dll");
        if (m) fn = (LogFn)GetProcAddress(m, "dgbridgeLogExternal");
    }
    if (!fn) return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    fn(buf);
}

// Sizes verified against the real optixu.1.dll (OptiX 3.0.0).
API RTresult rtuGetSizeForRTformat(unsigned format, size_t* size)
{
    static const unsigned char comp[] = { 4, 1, 1, 2, 2, 4, 4 };
    if (!size) return RT_ERROR_INVALID_VALUE;
    *size = 0;
    if (format > 0x100 && format < 0x11D) {
        unsigned i = format - 0x101;
        *size = (size_t)comp[i / 4] * (i % 4 + 1);
    }
    logLine("[optixu] rtuGetSizeForRTformat(0x%x) = %u", format, (unsigned)*size);
    return RT_SUCCESS;
}

// Names verified against the real optixu.1.dll (OptiX 3.0.0).
API RTresult rtuNameForType(unsigned type, char* buffer, RTsize bufferSize)
{
    static const char* names[] = {
        "unknown", "rtGroup", "rtGeometryGroup", "rtTransform", "rtSelector",
        "rtGeometryInstance", "rtBuffer", "rtTexturesampler", "rtObject",
        "matrix2x2", "matrix2x3", "matrix2x4", "matrix3x2", "matrix3x3", "matrix3x4",
        "matrix4x2", "matrix4x3", "matrix4x4", "float", "float2", "float3", "float4",
        "int", "int2", "int3", "int4", "uint", "uint2", "uint3", "uint4",
        "userdata", "rtProgram" };
    if (!buffer || !bufferSize) return RT_ERROR_INVALID_VALUE;
    const char* n = (type >= 0x200 && type <= 0x21F) ? names[type - 0x200] : "unknown";
    std::strncpy(buffer, n, bufferSize - 1);
    buffer[bufferSize - 1] = 0;
    return RT_SUCCESS;
}

#define UNSUPPORTED(name)                                                              \
    API RTresult name(void* a, ...)                                                    \
    {                                                                                  \
        logLine("[optixu] !!! " #name "(%p, ...) called - NOT SUPPORTED by DG Bridge", a); \
        return RT_ERROR_NOT_SUPPORTED;                                                 \
    }

UNSUPPORTED(rtuCUDACompileFile)
UNSUPPORTED(rtuCUDACompileString)
UNSUPPORTED(rtuCUDAGetCompileResult)
UNSUPPORTED(rtuCreateClusteredMesh)
UNSUPPORTED(rtuCreateClusteredMeshExt)
UNSUPPORTED(rtuTraversalCreate)
UNSUPPORTED(rtuTraversalDestroy)
UNSUPPORTED(rtuTraversalGetAccelData)
UNSUPPORTED(rtuTraversalGetAccelDataSize)
UNSUPPORTED(rtuTraversalGetErrorString)
UNSUPPORTED(rtuTraversalMapOutput)
UNSUPPORTED(rtuTraversalMapRays)
UNSUPPORTED(rtuTraversalMapResults)
UNSUPPORTED(rtuTraversalPreprocess)
UNSUPPORTED(rtuTraversalSetAccelData)
UNSUPPORTED(rtuTraversalSetMesh)
UNSUPPORTED(rtuTraversalSetOption)
UNSUPPORTED(rtuTraversalSetTriangles)
UNSUPPORTED(rtuTraversalTraverse)
UNSUPPORTED(rtuTraversalUnmapOutput)
UNSUPPORTED(rtuTraversalUnmapRays)
UNSUPPORTED(rtuTraversalUnmapResults)
