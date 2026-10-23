// dgbridge_protocol.h - shared between the 32-bit optix.1.dll stub and the
// 64-bit dgbridge_helper.exe. All structures use fixed-size types and explicit
// padding so they have the same layout in both processes.
//
// Transport: one named shared-memory block (header + data window) and two
// auto-reset events. The stub writes a command into the header, signals
// "req", and waits for "resp". Only one command is in flight at a time.
//
// Scene data is sent as a byte stream split into window-sized DGB_CHUNK
// commands and processed by DGB_SYNC. Stream layout: a sequence of records,
// each starting with a DgbRecord header, ending with DGB_REC_END.

#pragma once
#include <stdint.h>

#define DGB_MAGIC 0x44474252u          // 'DGBR'
#define DGB_PROTOCOL_VERSION 6u
#define DGB_SHM_HEADER_SIZE 4096u
#define DGB_SHM_DATA_SIZE (32u * 1024u * 1024u)

enum DgbCmd : uint32_t {
    DGB_HELLO = 1,   // -> msg = GPU name / init errors
    DGB_CHUNK = 2,   // dataSize bytes of scene stream in the data window
    DGB_SYNC = 3,    // process the accumulated stream
    DGB_RENDER = 4,  // render a frame (DgbRenderParams in the data window)
    DGB_READ = 5,    // arg[0] = offset, arg[1] = size: copy image bytes into the window
    DGB_QUIT = 6,
};

struct DgbHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t cmd;
    uint32_t status;      // 0 = ok, otherwise error (msg has details)
    uint64_t dataSize;
    uint64_t arg[8];
    char msg[512];
};
static_assert(sizeof(DgbHeader) == 600, "DgbHeader layout");

// ---- scene stream ------------------------------------------------------------

enum : uint32_t {
    DGB_REC_MESH = 0x4853454Du,       // 'MESH'
    DGB_REC_INSTANCES = 0x54534E49u,  // 'INST'
    DGB_REC_TEXTURE = 0x58455454u,    // 'TTEX'
    DGB_REC_END = 0x20444E45u,        // 'END '
};

struct DgbRecord {
    uint32_t tag;
    uint32_t reserved;
};

enum : uint32_t { DGB_MESH_NORMALS = 1, DGB_MESH_TEXCOORD0 = 2, DGB_MESH_TEXCOORD1 = 4, DGB_MESH_TANGENTS = 8 };

// Followed by: float pos[numVerts*3],
//              float nrm[numVerts*3]  (if DGB_MESH_NORMALS),
//              float tc0[numVerts*2]  (if DGB_MESH_TEXCOORD0),
//              float tc1[numVerts*2]  (if DGB_MESH_TEXCOORD1),
//              float tan[numVerts*3]  (if DGB_MESH_TANGENTS),
//              uint32 idx[numTris*3]
struct DgbMeshHeader {
    uint64_t key;
    uint32_t numVerts;
    uint32_t numTris;
    uint32_t flags;
    uint32_t pad;
};

// Followed by width*height*4 bytes of RGBA8 texels, row 0 first.
struct DgbTextureHeader {
    uint64_t key;
    uint32_t width;
    uint32_t height;
};
static_assert(sizeof(DgbTextureHeader) == 16, "DgbTextureHeader layout");
static_assert(sizeof(DgbMeshHeader) == 24, "DgbMeshHeader layout");

enum : uint32_t {
    DGB_MAT_PHONG = 0,
    DGB_MAT_CARPAINT = 1,
    DGB_MAT_METAL = 2,
    DGB_MAT_GLASS = 3,
    DGB_MAT_OTHER = 4,
};

// Texture sampler settings packed into DgbInstance::diffuseSampler:
//   bits 0-1: wrap mode U, bits 2-3: wrap mode V (OptiX 3 RTwrapmode:
//   0 repeat, 1 clamp to edge, 2 mirror, 3 clamp to border)
//   bit 4: linear filtering (else nearest)
//   bits 8-9: texture coordinate set (0 or 1)
#define DGB_SAMPLER_LINEAR 0x10u

// INSTANCES record: uint32 count, uint32 pad, then DgbInstance[count]
struct DgbInstance {
    uint64_t meshKey;
    float xform[12];      // row-major 3x4 object-to-world
    float base[3];        // base color
    float reflMin;        // reflectivity at normal incidence
    float reflMax;        // reflectivity at grazing angles
    uint32_t kind;        // DGB_MAT_*
    uint64_t diffuseTex;  // texture key, 0 = none
    uint32_t diffuseSampler;
    uint32_t pad0;
    float specular[3];    // highlight / reflection tint
    float shininess;      // Phong exponent of highlights and glossy reflections
    float base2[3];       // carpaint: color at grazing angles (diffuseColor1)
    float ior;            // glass: index of refraction
    // Phong extras (as in the demo's phong shader)
    uint32_t componentState;   // feature bits: 16 transparent texture, 128 alpha mode, 1024 bump map
    float transparency;
    float transparentColor[4];
    float bumpScale;
    uint32_t bumpSampler;      // DGB_SAMPLER_* bits (wrap, filter)
    uint64_t bumpTex;          // texture key, 0 = none
    uint64_t transparentTex;   // texture key, 0 = none
    uint32_t transparentSampler;
    uint32_t pad1;
    // carpaint
    float glossyColor[3];
    float glossiness;
    float reflectivityFalloff;
    // metal
    float shininessU;
    float shininessV;
    float absorption;
    // glass
    float reflectionColor[3];
    float pad2;
};
static_assert(sizeof(DgbInstance) == 232, "DgbInstance layout");

// ---- per-frame parameters --------------------------------------------------

struct DgbRenderParams {
    uint32_t width;
    uint32_t height;
    uint32_t frameNumber;
    uint32_t maxDepth;    // maxReflectionDepth
    float eye[3], U[3], V[3], W[3];
    float sunDir[3];
    float sunColor[3];    // as set by the demo (physical units)
    float epsilon;
    float overcast;       // 0 = clear sky, 1 = fully overcast
    float lensRadius;     // camLensRadius (0 = pinhole)
    float focalDistance;  // camFocalDistance
    float skyC[5][3];     // Preetham coefficients c0..c4
    float skyInvDivisorYxy[3];
};
static_assert(sizeof(DgbRenderParams) == 176, "DgbRenderParams layout");
