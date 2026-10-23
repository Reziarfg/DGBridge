// dgbridge_helper.cpp - DG Bridge stage 4b: 64-bit OptiX 9 render helper.
//
// Started by the 32-bit optix.1.dll stub inside Design Garage:
//   dgbridge_helper.exe <parent pid> "<log folder>"
// Talks to the stub through shared memory (see dgbridge_protocol.h):
//   - receives meshes and an instance/material table, builds hardware
//     acceleration structures (one GAS per mesh, one IAS for the scene)
//   - renders frames with the programs in dgbridge_device.ptx
//   - optionally runs the OptiX AI denoiser on the accumulated image
//     (DGBRIDGE_DENOISE=1)
//   - hands the float4 image back for the stub to copy into the demo's
//     OpenGL buffer
// Exits when the parent process goes away or on DGB_QUIT.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dgbridge_shared.h"

// ---------------------------------------------------------------------------
// Logging and error handling
// ---------------------------------------------------------------------------

static FILE* g_log = nullptr;

static void logf(const char* fmt, ...)
{
    if (!g_log) return;
    SYSTEMTIME st;
    GetLocalTime(&st);
    std::fprintf(g_log, "[%02d:%02d:%02d.%03d] ", st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_log, fmt, ap);
    va_end(ap);
    std::fputc('\n', g_log);
    std::fflush(g_log);
}

static std::string strf(const char* fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return buf;
}

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t e_ = (call);                                                           \
        if (e_ != cudaSuccess)                                                             \
            throw std::runtime_error(strf("%s failed: %s", #call, cudaGetErrorString(e_))); \
    } while (0)

static char g_optixLog[4096];
static size_t g_optixLogSize;

#define OPTIX_CHECK(call)                                                                  \
    do {                                                                                   \
        g_optixLogSize = sizeof(g_optixLog);                                               \
        OptixResult r_ = (call);                                                           \
        if (r_ != OPTIX_SUCCESS)                                                           \
            throw std::runtime_error(strf("%s failed: %s", #call, optixGetErrorString(r_))); \
    } while (0)

static void optixLogCallback(unsigned level, const char* tag, const char* msg, void*)
{
    logf("[OptiX %u][%s] %s", level, tag, msg);
}

// ---------------------------------------------------------------------------
// Device memory helper
// ---------------------------------------------------------------------------

struct DeviceBuffer {
    CUdeviceptr ptr = 0;
    size_t size = 0;
    void alloc(size_t bytes)
    {
        if (bytes <= size && ptr) return;
        free();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), std::max<size_t>(bytes, 16)));
        size = bytes;
    }
    void upload(const void* data, size_t bytes)
    {
        alloc(bytes);
        if (bytes) CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(ptr), data, bytes, cudaMemcpyHostToDevice));
    }
    void free()
    {
        if (ptr) cudaFree(reinterpret_cast<void*>(ptr));
        ptr = 0;
        size = 0;
    }
};

// ---------------------------------------------------------------------------
// Renderer state
// ---------------------------------------------------------------------------

struct Mesh {
    DeviceBuffer positions, normals, texcoords[2], tangents, indices, gas;
    OptixTraversableHandle handle = 0;
    unsigned numVerts = 0, numTris = 0;
    uint32_t flags = 0;
};

struct Texture {
    cudaArray_t array = nullptr;
    unsigned width = 0, height = 0;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptyRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    HitData data;
};

static OptixDeviceContext g_ctx = nullptr;
static OptixModule g_module = nullptr;
static OptixPipeline g_pipeline = nullptr;
static OptixProgramGroup g_pgRaygen = nullptr, g_pgMissRadiance = nullptr, g_pgMissShadow = nullptr,
                         g_pgHit = nullptr, g_pgShadowHit = nullptr;
static OptixShaderBindingTable g_sbt = {};
static DeviceBuffer g_raygenRecord, g_missRecords, g_hitRecords;
static std::map<uint64_t, Mesh> g_meshes;
static std::map<uint64_t, Texture> g_textures;
static std::map<std::pair<uint64_t, uint32_t>, cudaTextureObject_t> g_texObjects;
static DeviceBuffer g_ias, g_instanceBuffer;
static OptixTraversableHandle g_iasHandle = 0;
static DeviceBuffer g_accum, g_output, g_params, g_materials;
static DeviceBuffer g_accumAlbedo, g_accumNormal, g_albedo, g_normal, g_denoised;
static bool g_denoise = false;  // DGBRIDGE_DENOISE=1 turns it on
static unsigned g_spp = 4;      // paths per pixel per launch (DGBRIDGE_SPP, 1..64)
static OptixDenoiser g_denoiser = nullptr;
static DeviceBuffer g_denoiserState, g_denoiserScratch;
static unsigned g_denoiserW = 0, g_denoiserH = 0;
static std::vector<unsigned char> g_image;
static unsigned g_width = 0, g_height = 0, g_sample = 0;
static DgbRenderParams g_lastParams = {};
static bool g_sceneChanged = true;
static std::vector<unsigned char> g_stream;
static std::string g_exeDir;
static bool g_physical = false;   // DGBRIDGE_PHYSICAL=1: physically correct light balance

// ---------------------------------------------------------------------------
// OptiX AI denoiser (AOV model with albedo + world-space normal guides)
// ---------------------------------------------------------------------------

static void setupDenoiser(unsigned w, unsigned h)
{
    if (!g_denoiser) {
        OptixDenoiserOptions opts = {};
        opts.guideAlbedo = 1;
        opts.guideNormal = 1;
        opts.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
        OPTIX_CHECK(optixDenoiserCreate(g_ctx, OPTIX_DENOISER_MODEL_KIND_AOV, &opts, &g_denoiser));
    }
    OptixDenoiserSizes sizes = {};
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(g_denoiser, w, h, &sizes));
    g_denoiserState.alloc(sizes.stateSizeInBytes);
    g_denoiserScratch.alloc(sizes.withoutOverlapScratchSizeInBytes);
    OPTIX_CHECK(optixDenoiserSetup(g_denoiser, 0, w, h, g_denoiserState.ptr, sizes.stateSizeInBytes,
                                   g_denoiserScratch.ptr, sizes.withoutOverlapScratchSizeInBytes));
    g_denoiserW = w;
    g_denoiserH = h;
    logf("denoiser ready for %ux%u (state %zu KB, scratch %zu KB)", w, h, sizes.stateSizeInBytes >> 10,
         sizes.withoutOverlapScratchSizeInBytes >> 10);
}

static OptixImage2D image2D(const DeviceBuffer& b, unsigned w, unsigned h)
{
    OptixImage2D img = {};
    img.data = b.ptr;
    img.width = w;
    img.height = h;
    img.rowStrideInBytes = w * sizeof(float4);
    img.pixelStrideInBytes = sizeof(float4);
    img.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    return img;
}

static void denoise(unsigned w, unsigned h)
{
    if (g_denoiserW != w || g_denoiserH != h) setupDenoiser(w, h);
    OptixDenoiserParams dp = {};
    dp.blendFactor = 0.0f;   // intensity and average color are computed automatically
    OptixDenoiserGuideLayer guide = {};
    guide.albedo = image2D(g_albedo, w, h);
    guide.normal = image2D(g_normal, w, h);
    OptixDenoiserLayer layer = {};
    layer.input = image2D(g_output, w, h);
    layer.output = image2D(g_denoised, w, h);
    layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;
    OPTIX_CHECK(optixDenoiserInvoke(g_denoiser, 0, &dp, g_denoiserState.ptr, g_denoiserState.size, &guide, &layer, 1,
                                    0, 0, g_denoiserScratch.ptr, g_denoiserScratch.size));
}

static std::string readFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string initRenderer()
{
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) throw std::runtime_error("no CUDA device");
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));
    logf("GPU: %s (compute %d.%d, %zu MB)", prop.name, prop.major, prop.minor, prop.totalGlobalMem >> 20);

    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions ctxOptions = {};
    ctxOptions.logCallbackFunction = &optixLogCallback;
    ctxOptions.logCallbackLevel = 3;
    OPTIX_CHECK(optixDeviceContextCreate(nullptr, &ctxOptions, &g_ctx));

    const std::string ptx = readFile(g_exeDir + "\\dgbridge_device.ptx");

    OptixModuleCompileOptions mco = {};
    mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pco = {};
    pco.usesMotionBlur = 0;
    pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pco.numPayloadValues = 11;
    pco.numAttributeValues = 2;
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pco.pipelineLaunchParamsVariableName = "params";
    pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    OPTIX_CHECK(optixModuleCreate(g_ctx, &mco, &pco, ptx.c_str(), ptx.size(), g_optixLog, &g_optixLogSize, &g_module));

    OptixProgramGroupOptions pgo = {};
    OptixProgramGroupDesc d[5] = {};
    d[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    d[0].raygen.module = g_module;
    d[0].raygen.entryFunctionName = "__raygen__main";
    d[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    d[1].miss.module = g_module;
    d[1].miss.entryFunctionName = "__miss__radiance";
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    d[2].miss.module = g_module;
    d[2].miss.entryFunctionName = "__miss__shadow";
    d[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    d[3].hitgroup.moduleCH = g_module;
    d[3].hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    d[3].hitgroup.moduleAH = g_module;
    d[3].hitgroup.entryFunctionNameAH = "__anyhit__radiance";
    d[4].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    d[4].hitgroup.moduleAH = g_module;
    d[4].hitgroup.entryFunctionNameAH = "__anyhit__shadow";
    OptixProgramGroup groups[5] = {};
    OPTIX_CHECK(optixProgramGroupCreate(g_ctx, d, 5, &pgo, g_optixLog, &g_optixLogSize, groups));
    g_pgRaygen = groups[0];
    g_pgMissRadiance = groups[1];
    g_pgMissShadow = groups[2];
    g_pgHit = groups[3];
    g_pgShadowHit = groups[4];

    OptixPipelineLinkOptions plo = {};
    plo.maxTraceDepth = 1;
    OPTIX_CHECK(optixPipelineCreate(g_ctx, &pco, &plo, groups, 5, g_optixLog, &g_optixLogSize, &g_pipeline));

    EmptyRecord rg;
    OPTIX_CHECK(optixSbtRecordPackHeader(g_pgRaygen, &rg));
    g_raygenRecord.upload(&rg, sizeof(rg));
    EmptyRecord ms[2];
    OPTIX_CHECK(optixSbtRecordPackHeader(g_pgMissRadiance, &ms[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(g_pgMissShadow, &ms[1]));
    g_missRecords.upload(ms, sizeof(ms));

    g_sbt.raygenRecord = g_raygenRecord.ptr;
    g_sbt.missRecordBase = g_missRecords.ptr;
    g_sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    g_sbt.missRecordCount = 2;

    // Dummy hit records (radiance + shadow) so the SBT is valid before the first scene.
    HitRecord dummy[2] = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(g_pgHit, &dummy[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(g_pgShadowHit, &dummy[1]));
    g_hitRecords.upload(dummy, sizeof(dummy));
    g_sbt.hitgroupRecordBase = g_hitRecords.ptr;
    g_sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    g_sbt.hitgroupRecordCount = 2;

    g_params.alloc(sizeof(LaunchParams));
    MaterialGPU dummyMat = {};
    g_materials.upload(&dummyMat, sizeof(dummyMat));
    logf("renderer ready");
    return strf("%s (compute %d.%d, %zu MB), OptiX %d.%d.%d", prop.name, prop.major, prop.minor,
                prop.totalGlobalMem >> 20, OPTIX_VERSION / 10000, (OPTIX_VERSION % 10000) / 100, OPTIX_VERSION % 100);
}

// ---------------------------------------------------------------------------
// Scene building
// ---------------------------------------------------------------------------

static void buildAccel(const OptixBuildInput& input, DeviceBuffer& out, OptixTraversableHandle& handle)
{
    OptixAccelBuildOptions opts = {};
    opts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    opts.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(g_ctx, &opts, &input, 1, &sizes));
    DeviceBuffer temp;
    temp.alloc(sizes.tempSizeInBytes);
    out.alloc(sizes.outputSizeInBytes);
    OPTIX_CHECK(optixAccelBuild(g_ctx, 0, &opts, &input, 1, temp.ptr, sizes.tempSizeInBytes, out.ptr,
                                sizes.outputSizeInBytes, &handle, nullptr, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    temp.free();
}

struct StreamReader {
    const unsigned char* p;
    const unsigned char* end;
    const void* take(size_t n)
    {
        if ((size_t)(end - p) < n) throw std::runtime_error("scene stream truncated");
        const void* r = p;
        p += n;
        return r;
    }
};

static void addMesh(StreamReader& in)
{
    DgbMeshHeader mh;
    std::memcpy(&mh, in.take(sizeof(mh)), sizeof(mh));
    const size_t posBytes = (size_t)mh.numVerts * 12, tcBytes = (size_t)mh.numVerts * 8,
                 idxBytes = (size_t)mh.numTris * 12;
    const void* pos = in.take(posBytes);
    const void* nrm = (mh.flags & DGB_MESH_NORMALS) ? in.take(posBytes) : nullptr;
    const void* tc0 = (mh.flags & DGB_MESH_TEXCOORD0) ? in.take(tcBytes) : nullptr;
    const void* tc1 = (mh.flags & DGB_MESH_TEXCOORD1) ? in.take(tcBytes) : nullptr;
    const void* tan = (mh.flags & DGB_MESH_TANGENTS) ? in.take(posBytes) : nullptr;
    const void* idx = in.take(idxBytes);
    if (g_meshes.count(mh.key)) return;   // already have it
    if (mh.numVerts == 0 || mh.numTris == 0) return;

    Mesh& m = g_meshes[mh.key];
    m.numVerts = mh.numVerts;
    m.numTris = mh.numTris;
    m.flags = mh.flags;
    m.positions.upload(pos, posBytes);
    if (nrm) m.normals.upload(nrm, posBytes);
    if (tc0) m.texcoords[0].upload(tc0, tcBytes);
    if (tc1) m.texcoords[1].upload(tc1, tcBytes);
    if (tan) m.tangents.upload(tan, posBytes);
    m.indices.upload(idx, idxBytes);

    // Any-hit is switched off per instance for opaque materials; where it runs,
    // it must run once per triangle for stochastic transparency to be correct.
    const unsigned flags[1] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
    OptixBuildInput bi = {};
    bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    bi.triangleArray.vertexStrideInBytes = 12;
    bi.triangleArray.numVertices = mh.numVerts;
    bi.triangleArray.vertexBuffers = &m.positions.ptr;
    bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    bi.triangleArray.indexStrideInBytes = 12;
    bi.triangleArray.numIndexTriplets = mh.numTris;
    bi.triangleArray.indexBuffer = m.indices.ptr;
    bi.triangleArray.flags = flags;
    bi.triangleArray.numSbtRecords = 1;
    buildAccel(bi, m.gas, m.handle);
}

static void addTexture(StreamReader& in)
{
    DgbTextureHeader th;
    std::memcpy(&th, in.take(sizeof(th)), sizeof(th));
    const size_t bytes = (size_t)th.width * th.height * 4;
    const void* texels = in.take(bytes);
    if (g_textures.count(th.key) || !th.width || !th.height) return;
    Texture& t = g_textures[th.key];
    t.width = th.width;
    t.height = th.height;
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    CUDA_CHECK(cudaMallocArray(&t.array, &desc, th.width, th.height));
    CUDA_CHECK(cudaMemcpy2DToArray(t.array, 0, 0, texels, (size_t)th.width * 4, (size_t)th.width * 4, th.height,
                                   cudaMemcpyHostToDevice));
}

// One texture object per (texture, sampler settings) pair.
static cudaTextureObject_t textureObject(uint64_t key, uint32_t sampler)
{
    auto it = g_texObjects.find({ key, sampler & 0x1Fu });
    if (it != g_texObjects.end()) return it->second;
    auto t = g_textures.find(key);
    if (t == g_textures.end()) return 0;

    static const cudaTextureAddressMode modes[4] = { cudaAddressModeWrap, cudaAddressModeClamp,
                                                     cudaAddressModeMirror, cudaAddressModeBorder };
    cudaResourceDesc res = {};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = t->second.array;
    cudaTextureDesc td = {};
    td.addressMode[0] = modes[sampler & 3];
    td.addressMode[1] = modes[(sampler >> 2) & 3];
    td.filterMode = (sampler & DGB_SAMPLER_LINEAR) ? cudaFilterModeLinear : cudaFilterModePoint;
    td.readMode = cudaReadModeNormalizedFloat;
    td.normalizedCoords = 1;
    cudaTextureObject_t obj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&obj, &res, &td, nullptr));
    g_texObjects[{ key, sampler & 0x1Fu }] = obj;
    return obj;
}

static void setInstances(StreamReader& in)
{
    uint32_t count = *(const uint32_t*)in.take(4);
    in.take(4);
    std::vector<DgbInstance> src(count);
    if (count) std::memcpy(src.data(), in.take(sizeof(DgbInstance) * count), sizeof(DgbInstance) * count);

    std::vector<OptixInstance> instances;
    std::vector<HitRecord> records;
    std::vector<MaterialGPU> materials;
    unsigned missing = 0, textured = 0, seeThrough = 0, bumped = 0;
    for (const DgbInstance& s : src) {
        auto it = g_meshes.find(s.meshKey);
        if (it == g_meshes.end()) {
            ++missing;
            continue;
        }
        const Mesh& m = it->second;
        OptixInstance oi = {};
        std::memcpy(oi.transform, s.xform, sizeof(oi.transform));
        oi.instanceId = (unsigned)instances.size();
        oi.sbtOffset = (unsigned)records.size();   // two records per instance: radiance, shadow
        oi.visibilityMask = s.kind == DGB_MAT_GLASS ? MASK_GLASS : MASK_OPAQUE;
        // Any-hit only where the material can be (partly) see-through
        const uint32_t cs = s.componentState;
        bool seeThroughMat = false;
        if (s.kind == DGB_MAT_PHONG) {
            if (cs & CS_ALPHA_MODE)
                seeThroughMat = (cs & CS_TRANSPARENT_TEX) || s.transparentColor[3] * s.transparency < 1.0f;
            else
                seeThroughMat = (cs & CS_TRANSPARENT_TEX) || s.transparency * std::max(s.transparentColor[0],
                                    std::max(s.transparentColor[1], s.transparentColor[2])) > 0.0f;
        }
        // Glass also needs any-hit: shadow rays are tinted by refractionColor.
        oi.flags = (seeThroughMat || s.kind == DGB_MAT_GLASS) ? OPTIX_INSTANCE_FLAG_NONE
                                                              : OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
        if (seeThroughMat) ++seeThrough;
        oi.traversableHandle = m.handle;
        instances.push_back(oi);

        HitRecord r = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(g_pgHit, &r));
        r.data.positions = reinterpret_cast<const float3*>(m.positions.ptr);
        r.data.normals = (m.flags & DGB_MESH_NORMALS) ? reinterpret_cast<const float3*>(m.normals.ptr) : nullptr;
        const unsigned set = (s.diffuseSampler >> 8) & 1;
        const bool hasSet = (m.flags & (set ? DGB_MESH_TEXCOORD1 : DGB_MESH_TEXCOORD0)) != 0;
        r.data.texcoords = hasSet ? reinterpret_cast<const float2*>(m.texcoords[set].ptr) : nullptr;
        r.data.diffuseTex = (s.diffuseTex && hasSet) ? textureObject(s.diffuseTex, s.diffuseSampler) : 0;
        if (r.data.diffuseTex) ++textured;
        const bool hasTc1 = (m.flags & DGB_MESH_TEXCOORD1) != 0;
        r.data.texcoords1 = hasTc1 ? reinterpret_cast<const float2*>(m.texcoords[1].ptr) : nullptr;
        r.data.tangents = (m.flags & DGB_MESH_TANGENTS) ? reinterpret_cast<const float3*>(m.tangents.ptr) : nullptr;
        r.data.bumpTex = (s.bumpTex && hasTc1) ? textureObject(s.bumpTex, s.bumpSampler) : 0;
        r.data.transparentTex = (s.transparentTex && hasTc1) ? textureObject(s.transparentTex, s.transparentSampler) : 0;
        if (r.data.bumpTex && (s.componentState & CS_BUMP_TEX)) ++bumped;
        r.data.componentState = s.kind == DGB_MAT_PHONG ? s.componentState : 0;
        r.data.kind = s.kind;
        r.data.transparency = s.transparency;
        std::memcpy(r.data.transparentColor, s.transparentColor, sizeof(r.data.transparentColor));
        r.data.bumpScale = s.bumpScale;
        r.data.indices = reinterpret_cast<const uint3*>(m.indices.ptr);
        std::memcpy(r.data.base, s.base, sizeof(r.data.base));
        r.data.materialIndex = (unsigned)materials.size();
        records.push_back(r);
        HitRecord sr = r;                           // shadow record: same data, shadow any-hit program
        OPTIX_CHECK(optixSbtRecordPackHeader(g_pgShadowHit, &sr));
        records.push_back(sr);

        MaterialGPU mg = {};
        std::memcpy(mg.specular, s.specular, sizeof(mg.specular));
        mg.shininess = s.shininess;
        std::memcpy(mg.base2, s.base2, sizeof(mg.base2));
        mg.ior = s.ior;
        mg.reflMin = s.reflMin;
        mg.reflMax = s.reflMax;
        mg.kind = s.kind;
        mg.falloff = s.reflectivityFalloff;
        std::memcpy(mg.glossy, s.glossyColor, sizeof(mg.glossy));
        mg.glossiness = s.glossiness;
        mg.shinU = s.shininessU;
        mg.shinV = s.shininessV;
        mg.absorption = s.absorption;
        std::memcpy(mg.reflColor, s.reflectionColor, sizeof(mg.reflColor));
        materials.push_back(mg);
    }

    if (instances.empty()) {
        g_iasHandle = 0;
        logf("scene has no renderable instances (%u referenced unknown meshes)", missing);
        return;
    }

    g_instanceBuffer.upload(instances.data(), instances.size() * sizeof(OptixInstance));
    OptixBuildInput bi = {};
    bi.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    bi.instanceArray.instances = g_instanceBuffer.ptr;
    bi.instanceArray.numInstances = (unsigned)instances.size();
    buildAccel(bi, g_ias, g_iasHandle);

    g_hitRecords.upload(records.data(), records.size() * sizeof(HitRecord));
    g_materials.upload(materials.data(), materials.size() * sizeof(MaterialGPU));
    g_sbt.hitgroupRecordBase = g_hitRecords.ptr;
    g_sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    g_sbt.hitgroupRecordCount = (unsigned)records.size();

    unsigned counts[5] = {};
    for (const DgbInstance& s : src) counts[std::min<uint32_t>(s.kind, 4)]++;
    logf("scene: %u instances (%u missing meshes, %u textured, %u see-through, %u bump-mapped), %zu meshes and %zu textures cached; materials: phong %u, carpaint %u, metal %u, glass %u, other %u",
         (unsigned)instances.size(), missing, textured, seeThrough, bumped, g_meshes.size(), g_textures.size(), counts[0], counts[1],
         counts[2], counts[3], counts[4]);
}

static void processStream()
{
    StreamReader in{ g_stream.data(), g_stream.data() + g_stream.size() };
    unsigned meshesBefore = (unsigned)g_meshes.size(), texturesBefore = (unsigned)g_textures.size();
    for (;;) {
        DgbRecord rec;
        std::memcpy(&rec, in.take(sizeof(rec)), sizeof(rec));
        if (rec.tag == DGB_REC_END) break;
        if (rec.tag == DGB_REC_MESH) addMesh(in);
        else if (rec.tag == DGB_REC_TEXTURE) addTexture(in);
        else if (rec.tag == DGB_REC_INSTANCES) setInstances(in);
        else throw std::runtime_error(strf("unknown stream record 0x%08x", rec.tag));
    }
    logf("sync: %zu bytes, %u new meshes, %u new textures", g_stream.size(), (unsigned)g_meshes.size() - meshesBefore,
         (unsigned)g_textures.size() - texturesBefore);
    g_stream.clear();
    g_stream.shrink_to_fit();
    g_sceneChanged = true;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

// Everything except the frame counter must match for samples to keep accumulating.
static bool sameView(DgbRenderParams a, DgbRenderParams b)
{
    a.frameNumber = b.frameNumber = 0;
    return !std::memcmp(&a, &b, sizeof(a));
}

static size_t render(const DgbRenderParams& rp)
{
    if (rp.width == 0 || rp.height == 0 || rp.width > 16384 || rp.height > 16384)
        throw std::runtime_error(strf("bad image size %ux%u", rp.width, rp.height));

    if (rp.width != g_width || rp.height != g_height) {
        g_width = rp.width;
        g_height = rp.height;
        const size_t px = (size_t)g_width * g_height * sizeof(float4);
        g_accum.alloc(px);
        g_output.alloc(px);
        if (g_denoise) {
            g_accumAlbedo.alloc(px);
            g_accumNormal.alloc(px);
            g_albedo.alloc(px);
            g_normal.alloc(px);
            g_denoised.alloc(px);
        }
        g_sceneChanged = true;
        logf("image size %ux%u", g_width, g_height);
    }
    if (g_sceneChanged || !sameView(rp, g_lastParams)) g_sample = 0;
    g_sceneChanged = false;
    g_lastParams = rp;

    LaunchParams lp = {};
    lp.accum = reinterpret_cast<float4*>(g_accum.ptr);
    lp.output = reinterpret_cast<float4*>(g_output.ptr);
    if (g_denoise) {
        lp.accumAlbedo = reinterpret_cast<float4*>(g_accumAlbedo.ptr);
        lp.accumNormal = reinterpret_cast<float4*>(g_accumNormal.ptr);
        lp.albedo = reinterpret_cast<float4*>(g_albedo.ptr);
        lp.normal = reinterpret_cast<float4*>(g_normal.ptr);
    }
    lp.width = g_width;
    lp.height = g_height;
    lp.sample = g_sample;
    lp.spp = g_spp;
    std::memcpy(lp.eye, rp.eye, sizeof(lp.eye));
    std::memcpy(lp.U, rp.U, sizeof(lp.U));
    std::memcpy(lp.V, rp.V, sizeof(lp.V));
    std::memcpy(lp.W, rp.W, sizeof(lp.W));
    std::memcpy(lp.sunDir, rp.sunDir, sizeof(lp.sunDir));
    std::memcpy(lp.sunColor, rp.sunColor, sizeof(lp.sunColor));
    lp.epsilon = rp.epsilon;
    lp.materials = reinterpret_cast<const MaterialGPU*>(g_materials.ptr);
    lp.maxDepth = rp.maxDepth;
    lp.overcast = rp.overcast;
    lp.lensRadius = rp.lensRadius;
    lp.focalDistance = rp.focalDistance;
    std::memcpy(lp.skyC, rp.skyC, sizeof(lp.skyC));
    std::memcpy(lp.skyInvDivisorYxy, rp.skyInvDivisorYxy, sizeof(lp.skyInvDivisorYxy));
    // The demo's light balance (see dgbridge_device.cu) unless DGBRIDGE_PHYSICAL=1.
    // Physical sun: full angular radius instead of half -> 2pi(1-cos r) / 2pi(1-cos r/2).
    lp.diffuseBounceScale = g_physical ? 1.0f : 0.31830989f;
    lp.sunScale = g_physical ? 3.99999f : 1.0f;
    lp.handle = g_iasHandle;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(g_params.ptr), &lp, sizeof(lp), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(g_pipeline, 0, g_params.ptr, sizeof(LaunchParams), &g_sbt, g_width, g_height, 1));
    const size_t bytes = (size_t)g_width * g_height * sizeof(float4);
    const DeviceBuffer* result = &g_output;
    if (g_denoise) {
        try {
            denoise(g_width, g_height);
            result = &g_denoised;
        } catch (const std::exception& e) {
            logf("denoiser failed, continuing without it: %s", e.what());
            g_denoise = false;
        }
    }
    g_image.resize(bytes);
    CUDA_CHECK(cudaMemcpy(g_image.data(), reinterpret_cast<void*>(result->ptr), bytes, cudaMemcpyDeviceToHost));
    ++g_sample;
    return bytes;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::printf("dgbridge_helper is started automatically by Design Garage's optix.1.dll.\n");
        return 1;
    }
    const DWORD parentPid = (DWORD)std::strtoul(argv[1], nullptr, 10);
    const std::string logDir = argv[2];
    g_log = std::fopen((logDir + "\\helper_log.txt").c_str(), "w");
    {
        char v[8] = {};
        g_physical = GetEnvironmentVariableA("DGBRIDGE_PHYSICAL", v, sizeof(v)) > 0 && v[0] == '1';
        g_denoise = GetEnvironmentVariableA("DGBRIDGE_DENOISE", v, sizeof(v)) > 0 && v[0] == '1';
        if (GetEnvironmentVariableA("DGBRIDGE_SPP", v, sizeof(v)) > 0) {
            const unsigned long n = std::strtoul(v, nullptr, 10);
            g_spp = (unsigned)std::min<unsigned long>(std::max<unsigned long>(n, 1), 64);
        }
    }

    char exePath[MAX_PATH];
    GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    g_exeDir = exePath;
    g_exeDir = g_exeDir.substr(0, g_exeDir.find_last_of("\\/"));
    logf("DG Bridge helper (stage 4b), parent pid %lu, exe dir %s", parentPid, g_exeDir.c_str());
    logf("light balance: %s", g_physical ? "physically correct (DGBRIDGE_PHYSICAL=1)" : "as in the original demo");
    logf("denoiser: %s", g_denoise ? "on (DGBRIDGE_DENOISE=1)" : "off (DGBRIDGE_DENOISE=1 to turn on)");
    logf("paths per pixel per launch: %u (DGBRIDGE_SPP)", g_spp);

    char name[128];
    std::snprintf(name, sizeof(name), "Local\\DGBridge_%lu_shm", parentPid);
    HANDLE map = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name);
    std::snprintf(name, sizeof(name), "Local\\DGBridge_%lu_req", parentPid);
    HANDLE req = OpenEventA(SYNCHRONIZE | EVENT_MODIFY_STATE, FALSE, name);
    std::snprintf(name, sizeof(name), "Local\\DGBridge_%lu_resp", parentPid);
    HANDLE resp = OpenEventA(SYNCHRONIZE | EVENT_MODIFY_STATE, FALSE, name);
    HANDLE parent = OpenProcess(SYNCHRONIZE, FALSE, parentPid);
    if (!map || !req || !resp || !parent) {
        logf("could not open IPC objects (map %p req %p resp %p parent %p, error %lu)", map, req, resp, parent,
             GetLastError());
        return 1;
    }
    unsigned char* view = (unsigned char*)MapViewOfFile(map, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (!view) {
        logf("MapViewOfFile failed (%lu)", GetLastError());
        return 1;
    }
    DgbHeader* hdr = (DgbHeader*)view;
    unsigned char* data = view + DGB_SHM_HEADER_SIZE;

    bool initialized = false;
    unsigned long long frames = 0;
    for (;;) {
        HANDLE waits[2] = { req, parent };
        DWORD w = WaitForMultipleObjects(2, waits, FALSE, INFINITE);
        if (w != WAIT_OBJECT_0) {
            logf("parent process gone, exiting");
            break;
        }
        hdr->status = 0;
        hdr->msg[0] = 0;
        bool quit = false;
        try {
            if (hdr->magic != DGB_MAGIC || hdr->version != DGB_PROTOCOL_VERSION)
                throw std::runtime_error(strf("protocol mismatch (magic %08x version %u, expected version %u)",
                                              hdr->magic, hdr->version, DGB_PROTOCOL_VERSION));
            switch (hdr->cmd) {
            case DGB_HELLO: {
                const std::string info = initRenderer();
                initialized = true;
                std::snprintf(hdr->msg, sizeof(hdr->msg), "%s", info.c_str());
                break;
            }
            case DGB_CHUNK:
                if (hdr->dataSize > DGB_SHM_DATA_SIZE) throw std::runtime_error("chunk too large");
                g_stream.insert(g_stream.end(), data, data + hdr->dataSize);
                break;
            case DGB_SYNC:
                if (!initialized) throw std::runtime_error("not initialized");
                processStream();
                break;
            case DGB_RENDER: {
                if (!initialized) throw std::runtime_error("not initialized");
                DgbRenderParams rp;
                std::memcpy(&rp, data, sizeof(rp));
                hdr->arg[0] = render(rp);
                hdr->arg[1] = g_sample;
                if (++frames <= 3 || frames % 500 == 0)
                    logf("frame %llu: %ux%u, %u paths per pixel accumulated", frames, rp.width, rp.height,
                         g_sample * g_spp);
                break;
            }
            case DGB_READ: {
                const uint64_t off = hdr->arg[0], len = hdr->arg[1];
                if (len > DGB_SHM_DATA_SIZE || off + len > g_image.size()) throw std::runtime_error("bad read range");
                std::memcpy(data, g_image.data() + off, (size_t)len);
                hdr->dataSize = len;
                break;
            }
            case DGB_QUIT:
                quit = true;
                break;
            default:
                throw std::runtime_error(strf("unknown command %u", hdr->cmd));
            }
        } catch (const std::exception& e) {
            hdr->status = 1;
            std::snprintf(hdr->msg, sizeof(hdr->msg), "%s", e.what());
            logf("ERROR (cmd %u): %s", hdr->cmd, e.what());
            if (g_optixLog[0]) logf("OptiX log: %s", g_optixLog);
            g_stream.clear();
        }
        SetEvent(resp);
        if (quit) break;
    }
    logf("exiting");
    return 0;
}
