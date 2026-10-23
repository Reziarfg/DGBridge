// dgbridge_shared.h - structures shared by the helper's host code and the
// OptiX device programs (both 64-bit).

#pragma once
#include <optix_types.h>
#include <vector_types.h>

#include "dgbridge_protocol.h"

// Instance visibility masks
#define MASK_OPAQUE 1u
#define MASK_GLASS 2u

// Per-instance material parameters, indexed by the instance's SBT offset.
struct MaterialGPU {
    float specular[3];
    float shininess;
    float base2[3];
    float ior;
    float reflMin;
    float reflMax;
    unsigned kind;
    float falloff;        // carpaint reflectivityFalloff
    float glossy[3];      // carpaint glossyColor
    float glossiness;     // carpaint glossiness
    float shinU, shinV;   // metal anisotropic exponents
    float absorption;     // metal extinction coefficient
    float reflColor[3];   // glass reflectionColor
    float pad[2];
};

struct LaunchParams {
    float4* accum;        // running sum of samples (w = sample count)
    float4* output;       // averaged image handed back to the demo (or to the denoiser)
    float4* accumAlbedo;  // denoiser guides: running sums of first-hit albedo and
    float4* accumNormal;  // world-space normal
    float4* albedo;       // averaged guides
    float4* normal;
    const MaterialGPU* materials;
    unsigned width;
    unsigned height;
    unsigned sample;      // launches since the last reset (0 = first)
    unsigned maxDepth;
    unsigned spp;         // paths per pixel per launch (DGBRIDGE_SPP)
    unsigned pad0;
    float eye[3], U[3], V[3], W[3];
    float sunDir[3];
    float sunColor[3];
    float epsilon;
    float overcast;
    float lensRadius;
    float focalDistance;
    float skyC[5][3];
    float skyInvDivisorYxy[3];
    float diffuseBounceScale;   // 1/pi = the demo's weighting (default), 1 = physically correct
    float sunScale;             // 1 = the demo's sun solid angle (default), ~4 = physically correct
    OptixTraversableHandle handle;
};

// componentState bits used by the demo's phong shader
#define CS_DIFFUSE_TEX 4u
#define CS_TRANSPARENT_TEX 16u
#define CS_ALPHA_MODE 128u
#define CS_BUMP_TEX 1024u

// Per-instance shader binding table data (geometry and textures)
struct HitData {
    const float3* positions;
    const float3* normals;    // may be null: geometric normals are used then
    const float2* texcoords;  // set chosen for the diffuse texture, may be null
    const float2* texcoords1; // texcoord1 (bump and transparency maps), may be null
    const float3* tangents;   // may be null
    const uint3* indices;
    unsigned long long diffuseTex;       // cudaTextureObject_t, 0 = none
    unsigned long long bumpTex;
    unsigned long long transparentTex;
    float base[3];
    unsigned materialIndex;
    unsigned kind;            // DGB_MAT_*
    unsigned componentState;
    float transparency;
    float transparentColor[4];
    float bumpScale;
};
