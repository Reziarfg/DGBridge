// dgbridge_device.cu - OptiX 9 programs for DG Bridge stage 4b (several paths per launch).
//
// A progressive path tracer following Design Garage's own Monte Carlo
// renderer (montecarlo_ray_generation / montecarlo_miss / phong in the demo):
//   - sky: the Preetham model with the demo's coefficients (c0..c4,
//     inv_divisor_Yxy), Yxy -> RGB, scaled by 1/1000, blended with a
//     brightness-15 overcast sky by 'overcast' - same math as montecarlo_miss
//   - sun: sunColor * (1 - overcast)^2 * sun solid angle, sampled across the
//     sun's disk (soft shadows), as in the demo's phong shader
//   - light bounces between surfaces (diffuse and glossy); sky light arrives
//     through bounces that escape
//   - thin-lens depth of field from camLensRadius / camFocalDistance
//   - bump (normal) maps and transparency, ported from the demo's phong
//     shader: stochastic see-through for camera/bounce rays, colored
//     attenuation for shadow rays
//   - carpaint, metal and glass are ports of the demo's own shaders:
//       carpaint: mirror clear coat (specularColor, reflectivity with falloff)
//                 over diffuseColor0/1 + Blinn-Phong glossyColor layer
//       metal:    Ashikhmin-Shirley anisotropic lobe, conductor Fresnel
//       glass:    thin glass, Fresnel-weighted reflectionColor / refractionColor
//   - phong is still an approximation of the demo's phong shader
//   - light balance as in the demo: its diffuse bounces leave out the 1/pi of
//     the cosine-sampling density, so indirect diffuse light (and with it most
//     sky light) is pi times weaker than physically correct, and its sun solid
//     angle uses half the sun's angular radius. Both are reproduced by default
//     (diffuseBounceScale = 1/pi, sunScale = 1); DGBRIDGE_PHYSICAL=1 makes the
//     helper use physically correct values instead.
//   - albedo and normal of the first visible non-glass surface are accumulated
//     as guide images for the OptiX AI denoiser
//
// The closest-hit program only reports surface data; all shading happens in
// the ray generation program, so the trace depth stays at 1.

#include <optix.h>

#include "dgbridge_shared.h"

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

static __forceinline__ __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __forceinline__ __device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __forceinline__ __device__ float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
static __forceinline__ __device__ float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __forceinline__ __device__ float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
static __forceinline__ __device__ float3 operator*(float s, float3 a) { return a * s; }
static __forceinline__ __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static __forceinline__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static __forceinline__ __device__ float length(float3 a) { return sqrtf(dot(a, a)); }
static __forceinline__ __device__ float3 normalize(float3 a) { return a * rsqrtf(fmaxf(dot(a, a), 1e-30f)); }
static __forceinline__ __device__ float3 lerp(float3 a, float3 b, float t) { return a + (b - a) * t; }
static __forceinline__ __device__ float3 reflect(float3 d, float3 n) { return d - n * (2.0f * dot(d, n)); }
static __forceinline__ __device__ float3 f3(const float* p) { return make_float3(p[0], p[1], p[2]); }
static __forceinline__ __device__ float maxc(float3 a) { return fmaxf(a.x, fmaxf(a.y, a.z)); }
static __forceinline__ __device__ float3 expf3(float3 a) { return make_float3(expf(a.x), expf(a.y), expf(a.z)); }

static __forceinline__ __device__ unsigned tea(unsigned v0, unsigned v1)
{
    unsigned s0 = 0;
    for (int n = 0; n < 4; ++n) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}
static __forceinline__ __device__ float rnd(unsigned& seed)
{
    seed = 1664525u * seed + 1013904223u;
    return (float)(seed & 0x00FFFFFFu) / (float)0x01000000u;
}

// Orthonormal basis around n (Duff et al. 2017)
static __forceinline__ __device__ void onb(float3 n, float3& t, float3& b)
{
    const float s = copysignf(1.0f, n.z);
    const float a = -1.0f / (s + n.z);
    const float c = n.x * n.y * a;
    t = make_float3(1.0f + s * n.x * n.x * a, s * c, -s * n.x);
    b = make_float3(c, s + n.y * n.y * a, -n.y);
}

// Direction around an axis, given the cosine of the angle from the axis
static __forceinline__ __device__ float3 aroundAxis(float3 axis, float cosT, float phi)
{
    float3 t, b;
    onb(axis, t, b);
    const float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));
    return normalize(t * (cosf(phi) * sinT) + b * (sinf(phi) * sinT) + axis * cosT);
}
static __forceinline__ __device__ float3 sampleCosine(float3 n, unsigned& seed)
{
    const float u = rnd(seed), v = rnd(seed);
    return aroundAxis(n, sqrtf(1.0f - u), 6.2831853f * v);
}
static __forceinline__ __device__ float3 samplePhongLobe(float3 axis, float exponent, unsigned& seed)
{
    const float u = rnd(seed), v = rnd(seed);
    return aroundAxis(axis, powf(u, 1.0f / (exponent + 1.0f)), 6.2831853f * v);
}

extern "C" {
__constant__ LaunchParams params;
}

// ---------------------------------------------------------------------------
// Sky and sun, following the demo's montecarlo_miss and phong programs
// ---------------------------------------------------------------------------

// The demo's sun disk: half of acos(94 / 94.0011) radians in angular radius.
#define SUN_COS_HALF_ANGLE 0.99999707f
#define SUN_SOLID_ANGLE (6.2831853f * (1.0f - SUN_COS_HALF_ANGLE))

static __forceinline__ __device__ float3 skyRadiance(float3 dir)
{
    const float overcast = params.overcast;
    float3 sunlit = make_float3(0.0f, 0.0f, 0.0f);
    if (overcast < 1.0f) {
        float3 d = dir;
        float invUp = 1.0f / d.y;
        if (invUp < 0.0f) {   // below the horizon: mirror into the upper hemisphere
            d.y = -d.y;
            invUp = -invUp;
        }
        const float gamma = dot(f3(params.sunDir), d);
        const float acosGamma = acosf(fminf(fmaxf(gamma, -1.0f), 1.0f));
        const float3 c0 = f3(params.skyC[0]), c1 = f3(params.skyC[1]), c2 = f3(params.skyC[2]),
                     c3 = f3(params.skyC[3]), c4 = f3(params.skyC[4]);
        const float3 one = make_float3(1.0f, 1.0f, 1.0f);
        float3 Yxy = (one + c0 * expf3(c1 * invUp)) * (one + c2 * expf3(c3 * acosGamma) + c4 * (gamma * gamma));
        Yxy = Yxy * f3(params.skyInvDivisorYxy);
        Yxy.y = 0.33f + 1.2f * (Yxy.y - 0.33f);
        Yxy.z = 0.33f + 1.2f * (Yxy.z - 0.33f);
        // Yxy -> XYZ -> linear RGB, scaled to the demo's units
        const float Y = Yxy.x;
        const float X = Yxy.y * (Y / Yxy.z);
        const float Z = (1.0f - Yxy.y - Yxy.z) * (Y / Yxy.z);
        sunlit = make_float3(3.241f * X - 1.5374f * Y - 0.4986f * Z, -0.9692f * X + 1.876f * Y + 0.0416f * Z,
                             0.0556f * X - 0.204f * Y + 1.057f * Z) *
                 0.001f;
    }
    const float o = (1.0f + 2.0f * fabsf(dir.y)) / 3.0f * 15.0f;
    return lerp(sunlit, make_float3(o, o, o), overcast);
}

// Sun irradiance at normal incidence (before the BRDF and cosine terms)
static __forceinline__ __device__ float3 sunIrradiance()
{
    const float k = (1.0f - params.overcast) * (1.0f - params.overcast) * SUN_SOLID_ANGLE * params.sunScale;
    return f3(params.sunColor) * k;
}

// ---------------------------------------------------------------------------
// Tracing
// ---------------------------------------------------------------------------

struct Hit {
    float t;           // < 0: miss
    float3 n, ng, base;
    unsigned material;
};

static __forceinline__ __device__ Hit traceRadiance(float3 origin, float3 dir, float tmin)
{
    unsigned p0 = __float_as_uint(-1.0f), p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0, p8 = 0,
             p9 = 0, p10 = 0;
    optixTrace(params.handle, origin, dir, tmin, 1e30f, 0.0f, OptixVisibilityMask(MASK_OPAQUE | MASK_GLASS),
               OPTIX_RAY_FLAG_NONE, 0, 2, 0, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
    Hit h;
    h.t = __uint_as_float(p0);
    h.n = make_float3(__uint_as_float(p1), __uint_as_float(p2), __uint_as_float(p3));
    h.ng = make_float3(__uint_as_float(p4), __uint_as_float(p5), __uint_as_float(p6));
    h.base = make_float3(__uint_as_float(p7), __uint_as_float(p8), __uint_as_float(p9));
    h.material = p10;
    return h;
}

// Light reaching 'origin' along 'dir': 0 if blocked, dimmed by partly
// transparent surfaces and tinted by glass (refractionColor) in between.
static __forceinline__ __device__ float3 traceShadow(float3 origin, float3 dir)
{
    unsigned a0 = __float_as_uint(1.0f), a1 = a0, a2 = a0, visible = 0;
    optixTrace(params.handle, origin, dir, 0.0f, 1e30f, 0.0f, OptixVisibilityMask(MASK_OPAQUE | MASK_GLASS),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 1, 2, 1, a0, a1, a2,
               visible);
    if (!visible) return make_float3(0.0f, 0.0f, 0.0f);
    return make_float3(__uint_as_float(a0), __uint_as_float(a1), __uint_as_float(a2));
}

static __forceinline__ __device__ float fresnelBlend(float cosTheta, float rMin, float rMax)
{
    const float m = fminf(fmaxf(1.0f - cosTheta, 0.0f), 1.0f);
    const float m2 = m * m;
    return rMin + (rMax - rMin) * (m2 * m2 * m);
}

// ---------------------------------------------------------------------------
// One light path through pixel 'idx'. 'sampleIndex' counts all paths traced
// for this pixel since the last reset (0 = first, through the pixel center).
// ---------------------------------------------------------------------------

static __forceinline__ __device__ void tracePath(const uint3 idx, const unsigned sampleIndex, float3& radiance,
                                                 float3& gAlbedo, float3& gNormal)
{
    const unsigned w = params.width, h = params.height;
    const unsigned pixel = idx.y * w + idx.x;
    unsigned seed = tea(pixel, sampleIndex);

    // Camera: pinhole direction through the jittered pixel, then a thin lens
    const float jx = sampleIndex == 0 ? 0.5f : rnd(seed);
    const float jy = sampleIndex == 0 ? 0.5f : rnd(seed);
    const float dx = ((float)idx.x + jx) / (float)w * 2.0f - 1.0f;
    const float dy = ((float)idx.y + jy) / (float)h * 2.0f - 1.0f;
    const float3 eye = f3(params.eye), U = f3(params.U), V = f3(params.V), W = f3(params.W);
    float3 dir = normalize(dx * U + dy * V + W);
    float3 origin = eye;
    if (params.lensRadius > 0.0f && params.focalDistance > 0.0f && sampleIndex > 0) {
        const float focusT = params.focalDistance / fmaxf(dot(dir, normalize(W)), 1e-4f);
        const float3 focus = eye + dir * focusT;
        const float r = params.lensRadius * sqrtf(rnd(seed)), phi = 6.2831853f * rnd(seed);
        origin = eye + normalize(U) * (r * cosf(phi)) + normalize(V) * (r * sinf(phi));
        dir = normalize(focus - origin);
    }

    radiance = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    // Denoiser guides from the first visible surface (looking through glass)
    gAlbedo = make_float3(0.0f, 0.0f, 0.0f);
    gNormal = make_float3(0.0f, 0.0f, 0.0f);
    bool guideSet = false;

    if (params.handle == 0) {
        radiance = skyRadiance(dir);
    } else {
        const float eps = params.epsilon;
        const float3 sunDir = f3(params.sunDir);
        const float3 sunE = sunIrradiance();
        const bool sunOn = maxc(sunE) > 0.0f;
        const unsigned maxBounces = max(2u, min(params.maxDepth + 1u, 12u));
        unsigned bounces = 0, passes = 0;
        float tmin = 0.0f;

        for (;;) {
            const Hit hit = traceRadiance(origin, dir, tmin);
            if (hit.t < 0.0f) {
                const float3 sky = skyRadiance(dir);
                radiance = radiance + throughput * sky;
                if (!guideSet) {
                    gAlbedo = make_float3(fminf(sky.x / 15.0f, 1.0f), fminf(sky.y / 15.0f, 1.0f), fminf(sky.z / 15.0f, 1.0f));
                    guideSet = true;
                }
                break;
            }
            const MaterialGPU m = params.materials[hit.material];
            if (!guideSet && m.kind != DGB_MAT_GLASS) {
                guideSet = true;
                gNormal = hit.n;
                if (m.kind == DGB_MAT_METAL) gAlbedo = hit.base;
                else if (m.kind == DGB_MAT_CARPAINT) gAlbedo = hit.base + f3(m.glossy) + f3(m.specular) * m.reflMax;
                else gAlbedo = hit.base + f3(m.specular) * m.reflMax;
                gAlbedo = make_float3(fminf(gAlbedo.x, 1.0f), fminf(gAlbedo.y, 1.0f), fminf(gAlbedo.z, 1.0f));
            }
            const float3 P = origin + dir * hit.t;
            const float3 Vd = -dir;
            const float cosV = fmaxf(dot(hit.n, Vd), 0.0f);

            if (m.kind == DGB_MAT_GLASS) {
                // Demo's glass: exact dielectric Fresnel (always treated as entering),
                // reflection weighted by reflectionColor * F, transmission straight
                // through weighted by refractionColor; one is picked at random.
                const float ior = m.ior > 0.0f ? m.ior : 1.5f;
                const float cosI = fabsf(dot(hit.n, dir));
                const float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosI * cosI)) / ior;
                float F = 1.0f;
                if (sinT <= 1.0f) {
                    const float cosT = sqrtf(fmaxf(0.0f, 1.0f - sinT * sinT));
                    const float rs = (cosI * ior - cosT) / (cosI * ior + cosT);
                    const float rp = (cosI - cosT * ior) / (cosI + cosT * ior);
                    F = fminf((rs * rs + rp * rp) * 0.5f, 1.0f);
                }
                const float3 reflW = f3(m.reflColor) * F;
                const float3 transW = hit.base;   // refractionColor
                const float sr = reflW.x + reflW.y + reflW.z, st = transW.x + transW.y + transW.z;
                if (sr + st <= 0.0f) break;
                const float pR = sr / (sr + st);
                if (rnd(seed) < pR) {
                    if (++bounces > maxBounces) break;
                    throughput = throughput * reflW * (1.0f / pR);
                    dir = reflect(dir, hit.n);
                    origin = P + hit.ng * eps;
                    tmin = 0.0f;
                } else {
                    if (++passes > 8) break;
                    throughput = throughput * transW * (1.0f / (1.0f - pR));
                    origin = P;
                    tmin = eps;
                }
                continue;
            }

            const float3 R = reflect(dir, hit.n);

            if (m.kind == DGB_MAT_CARPAINT) {
                // Demo's carpaint: color flop diffuseColor0 -> diffuseColor1 toward
                // grazing angles, clear-coat reflectivity with falloff, diffuse +
                // glossy normalized to at most 1, three-way lobe selection.
                const float c = cosV;
                float3 diff = hit.base * c + f3(m.base2) * (1.0f - c);
                const float e = 1.0f + 4.0f * m.falloff;
                const float refl = fmaxf(fminf(m.reflMin + (m.reflMax - m.reflMin) * powf(fmaxf(1.0f - c, 0.0f), e),
                                               m.reflMax), m.reflMin);
                float3 gl = f3(m.glossy);
                const float sc = fminf(1.0f / fmaxf(maxc(diff + gl), 1e-6f), 1.0f);
                diff = diff * sc;
                gl = gl * sc;
                const float nExp = 1.0f + 100.0f * m.glossiness;
                const float3 spec = f3(m.specular);
                const float pb = (maxc(gl) + maxc(diff)) * (1.0f - refl);
                const float ps = maxc(spec) * refl;

                if (sunOn) {
                    const float3 L = aroundAxis(sunDir, 1.0f - rnd(seed) * (1.0f - SUN_COS_HALF_ANGLE),
                                                6.2831853f * rnd(seed));
                    const float ndl = dot(hit.n, L);
                    if (ndl > 0.0f && dot(hit.ng, L) > 0.0f) {
                        const float3 vis = traceShadow(P + hit.ng * eps, L);
                        if (maxc(vis) > 0.0f) {
                            const float3 H = normalize(L + Vd);
                            const float ndh = fmaxf(dot(hit.n, H), 0.0f);
                            const float g = fminf((nExp + 2.0f) * 0.15915494f * powf(ndh, nExp) /
                                                      (4.0f * fmaxf(fmaxf(ndl, cosV), 1e-3f)),
                                                  3.1830989f);
                            const float3 brdf = (diff * 0.31830989f + gl * g) * (1.0f - refl);
                            radiance = radiance + throughput * brdf * sunE * vis * ndl;
                        }
                    }
                }
                if (++bounces > maxBounces || ps + pb <= 0.0f) break;
                const float pS = ps / (ps + pb);
                if (rnd(seed) < pS) {
                    dir = R;                                           // clear coat mirror
                    throughput = throughput * spec * (refl / pS);
                } else {
                    const float pB = 1.0f - pS;
                    const float md = maxc(diff), mg = maxc(gl);
                    const float pd = md / fmaxf(md + mg, 1e-6f);
                    if (rnd(seed) < pd) {
                        dir = sampleCosine(hit.n, seed);
                        throughput = throughput * diff * ((1.0f - refl) / (pB * pd) * params.diffuseBounceScale);
                    } else {
                        const float3 Hs = samplePhongLobe(hit.n, nExp, seed);   // Blinn-Phong half vector
                        dir = reflect(dir, Hs);
                        throughput = throughput * gl * ((1.0f - refl) / (pB * (1.0f - pd)));
                    }
                }
                if (dot(dir, hit.ng) <= 0.0f) break;
            } else if (m.kind == DGB_MAT_METAL) {
                // Demo's metal: Ashikhmin-Shirley anisotropic lobe (shininess_u/v)
                // around a tangent built from the normal; conductor Fresnel with
                // n from the brightness of specularColor and k = absorption.
                const float3 n = hit.n;
                float3 t = fabsf(n.x) > fabsf(n.z) ? make_float3(-n.y, n.x, 0.0f) : make_float3(0.0f, -n.z, n.y);
                t = normalize(t);
                const float3 bt = cross(n, t);
                const float nu = fmaxf(m.shinU, 0.0f), nv = fmaxf(m.shinV, 0.0f);
                const float3 spec = f3(m.specular);
                const float lum = 0.3f * spec.x + 0.59f * spec.y + 0.11f * spec.z;
                const float sl = sqrtf(fmaxf(lum, 0.0f));
                const float etaC = (1.0f - sl) / (1.0f + sl);
                const float a = m.absorption * m.absorption + etaC * etaC;
                auto conductorF = [&](float c) {
                    const float c2 = c * c, tc = 2.0f * c * etaC;
                    const float rs = (c2 + a - tc) / (c2 + a + tc);
                    const float rp = (c2 * a - tc + 1.0f) / (c2 * a + tc + 1.0f);
                    return 0.5f * (rs + rp);
                };
                auto lobeExp = [&](float3 H) {
                    const float ht = dot(H, t), hb = dot(H, bt), hn = dot(H, n);
                    const float s2 = fmaxf(1.0f - hn * hn, 1e-8f);
                    return (nu * ht * ht + nv * hb * hb) / s2;
                };

                if (sunOn) {
                    const float3 L = aroundAxis(sunDir, 1.0f - rnd(seed) * (1.0f - SUN_COS_HALF_ANGLE),
                                                6.2831853f * rnd(seed));
                    const float ndl = dot(n, L);
                    if (ndl > 0.0f && dot(hit.ng, L) > 0.0f) {
                        const float3 vis = traceShadow(P + hit.ng * eps, L);
                        if (maxc(vis) > 0.0f) {
                            const float3 H = normalize(L + Vd);
                            const float hk = fmaxf(dot(H, Vd), 1e-4f);
                            const float d = sqrtf((nu + 1.0f) * (nv + 1.0f)) * 0.039788736f *   // 1/(8 pi)
                                            powf(fmaxf(dot(n, H), 0.0f), lobeExp(H)) /
                                            (hk * fmaxf(fmaxf(ndl, cosV), 1e-3f));
                            const float3 brdf = spec * (conductorF(hk) * fminf(d, 3.1830989f));
                            radiance = radiance + throughput * brdf * sunE * vis * ndl;
                        }
                    }
                }
                if (++bounces > maxBounces) break;
                // Sample the half vector (Ashikhmin & Shirley 2000)
                const float u1 = rnd(seed), u2 = rnd(seed);
                float phi;
                const float k = sqrtf((nu + 1.0f) / fmaxf(nv + 1.0f, 1e-6f));
                const float q = u1 < 0.25f ? 4.0f * u1 : u1 < 0.5f ? 4.0f * (0.5f - u1) : u1 < 0.75f ? 4.0f * (u1 - 0.5f)
                                                                                                   : 4.0f * (1.0f - u1);
                phi = atanf(k * tanf(q * 1.5707963f));
                if (u1 >= 0.25f && u1 < 0.5f) phi = 3.1415927f - phi;
                else if (u1 >= 0.5f && u1 < 0.75f) phi = 3.1415927f + phi;
                else if (u1 >= 0.75f) phi = 6.2831853f - phi;
                const float cp = cosf(phi), sp = sinf(phi);
                const float cosT = powf(fmaxf(u2, 1e-12f), 1.0f / (nu * cp * cp + nv * sp * sp + 1.0f));
                const float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));
                const float3 H = normalize(t * (cp * sinT) + bt * (sp * sinT) + n * cosT);
                dir = reflect(dir, H);
                const float ndl = dot(n, dir);
                if (ndl <= 0.0f || dot(dir, hit.ng) <= 0.0f) break;
                const float hk = fmaxf(dot(H, Vd), 1e-4f);
                throughput = throughput * spec * (conductorF(hk) * ndl / fmaxf(fmaxf(ndl, cosV), 1e-3f));
            } else {
                // Phong (and anything else): approximation of the demo's phong shader
                const float3 diffuse = hit.base;
                const float3 spec = f3(m.specular);
                const float exponent = fmaxf(m.shininess, 1.0f);
                const float pReflect =
                    m.kind == DGB_MAT_PHONG ? fminf(fmaxf(fresnelBlend(cosV, m.reflMin, m.reflMax), 0.0f), 1.0f) : 0.0f;

                if (sunOn) {
                    const float3 L = aroundAxis(sunDir, 1.0f - rnd(seed) * (1.0f - SUN_COS_HALF_ANGLE),
                                                6.2831853f * rnd(seed));
                    const float ndl = dot(hit.n, L);
                    if (ndl > 0.0f && dot(hit.ng, L) > 0.0f) {
                        const float3 vis = traceShadow(P + hit.ng * eps, L);
                        if (maxc(vis) > 0.0f) {
                            const float sp = powf(fmaxf(dot(R, L), 0.0f), exponent) * (exponent + 2.0f) * 0.15915494f;
                            const float3 brdf =
                                diffuse * (0.31830989f * (1.0f - pReflect)) + spec * fminf(sp, 3.1830989f);
                            radiance = radiance + throughput * brdf * sunE * vis * ndl;
                        }
                    }
                }
                if (++bounces > maxBounces) break;
                if (rnd(seed) < pReflect) {
                    dir = exponent >= 10000.0f ? R : samplePhongLobe(R, exponent, seed);
                } else {
                    if (maxc(diffuse) <= 0.0f) break;
                    dir = sampleCosine(hit.n, seed);
                    throughput = throughput * diffuse * params.diffuseBounceScale;
                }
                if (dot(dir, hit.ng) <= 0.0f) break;
            }
            origin = P + hit.ng * eps;
            tmin = 0.0f;

            // Russian roulette after a few bounces
            if (bounces >= 3) {
                const float q = fminf(fmaxf(maxc(throughput), 0.05f), 0.95f);
                if (rnd(seed) >= q) break;
                throughput = throughput * (1.0f / q);
            }
        }
    }

    if (!isfinite(radiance.x) || !isfinite(radiance.y) || !isfinite(radiance.z))
        radiance = make_float3(0.0f, 0.0f, 0.0f);
}

// ---------------------------------------------------------------------------
// Ray generation: params.spp paths per pixel per launch, accumulated over
// launches (the demo's own renderer traces one path per pixel per launch)
// ---------------------------------------------------------------------------

extern "C" __global__ void __raygen__main()
{
    const uint3 idx = optixGetLaunchIndex();
    const unsigned pixel = idx.y * params.width + idx.x;
    const unsigned spp = max(params.spp, 1u);

    float3 sum = make_float3(0.0f, 0.0f, 0.0f), sumA = sum, sumN = sum;
    for (unsigned s = 0; s < spp; ++s) {
        float3 radiance, gAlbedo, gNormal;
        tracePath(idx, params.sample * spp + s, radiance, gAlbedo, gNormal);
        sum = sum + radiance;
        sumA = sumA + gAlbedo;
        sumN = sumN + gNormal;
    }

    float4 acc = make_float4(sum.x, sum.y, sum.z, (float)spp);
    float4 accA = make_float4(sumA.x, sumA.y, sumA.z, 0.0f);
    float4 accN = make_float4(sumN.x, sumN.y, sumN.z, 0.0f);
    if (params.sample > 0) {
        const float4 prev = params.accum[pixel];
        acc = make_float4(prev.x + acc.x, prev.y + acc.y, prev.z + acc.z, prev.w + acc.w);
        if (params.accumAlbedo) {
            const float4 pa = params.accumAlbedo[pixel], pn = params.accumNormal[pixel];
            accA = make_float4(pa.x + accA.x, pa.y + accA.y, pa.z + accA.z, 0.0f);
            accN = make_float4(pn.x + accN.x, pn.y + accN.y, pn.z + accN.z, 0.0f);
        }
    }
    params.accum[pixel] = acc;
    const float inv = 1.0f / acc.w;
    params.output[pixel] = make_float4(acc.x * inv, acc.y * inv, acc.z * inv, 1.0f);
    if (params.accumAlbedo) {
        params.accumAlbedo[pixel] = accA;
        params.accumNormal[pixel] = accN;
        params.albedo[pixel] = make_float4(accA.x * inv, accA.y * inv, accA.z * inv, 0.0f);
        const float3 nn = make_float3(accN.x, accN.y, accN.z);
        const float nl = length(nn);
        params.normal[pixel] = nl > 1e-6f ? make_float4(nn.x / nl, nn.y / nl, nn.z / nl, 0.0f)
                                          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

extern "C" __global__ void __miss__radiance()
{
    optixSetPayload_0(__float_as_uint(-1.0f));
}

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_3(1u);
}

// ---------------------------------------------------------------------------
// Surface helpers shared by the hit programs
// ---------------------------------------------------------------------------

static __forceinline__ __device__ float2 interp2(const float2* a, uint3 tri, float2 b)
{
    const float2 t0 = a[tri.x], t1 = a[tri.y], t2 = a[tri.z];
    const float w0 = 1.0f - b.x - b.y;
    return make_float2(t0.x * w0 + t1.x * b.x + t2.x * b.y, t0.y * w0 + t1.y * b.x + t2.y * b.y);
}

// Opacity per channel, as computed by the demo's phong any-hit programs:
//   alpha mode (bit 128): transparency * alpha (texture alpha or transparentColor.w)
//   color mode:           1 - transparency * color (texture rgb or transparentColor.rgb)
// The transparency map is read with texcoord1, like the original shader.
static __forceinline__ __device__ float3 surfaceOpacity(const HitData* hd)
{
    const unsigned cs = hd->componentState;
    float4 tc = make_float4(hd->transparentColor[0], hd->transparentColor[1], hd->transparentColor[2],
                            hd->transparentColor[3]);
    if ((cs & CS_TRANSPARENT_TEX) && hd->transparentTex && hd->texcoords1) {
        const uint3 tri = hd->indices[optixGetPrimitiveIndex()];
        const float2 uv = interp2(hd->texcoords1, tri, optixGetTriangleBarycentrics());
        tc = tex2D<float4>(hd->transparentTex, uv.x, uv.y);
    }
    if (cs & CS_ALPHA_MODE) {
        const float o = tc.w * hd->transparency;
        return make_float3(o, o, o);
    }
    return make_float3(1.0f - hd->transparency * tc.x, 1.0f - hd->transparency * tc.y,
                       1.0f - hd->transparency * tc.z);
}

// Camera and bounce rays: pass through with probability 1 - opacity.
extern "C" __global__ void __anyhit__radiance()
{
    const HitData* hd = reinterpret_cast<const HitData*>(optixGetSbtDataPointer());
    if (hd->kind == DGB_MAT_GLASS) return;   // glass is handled in the ray generation program
    const float o = maxc(surfaceOpacity(hd));
    if (o >= 1.0f) return;
    const uint3 li = optixGetLaunchIndex();
    unsigned seed = tea(li.y * params.width + li.x, params.sample * 7919u + optixGetPrimitiveIndex() * 31u +
                                                       optixGetInstanceId() + __float_as_uint(optixGetRayTmax()));
    if (o < rnd(seed)) optixIgnoreIntersection();
}

// Shadow rays: dim the light by (1 - opacity); stop once it is nearly black.
extern "C" __global__ void __anyhit__shadow()
{
    const HitData* hd = reinterpret_cast<const HitData*>(optixGetSbtDataPointer());
    // Glass tints the light by refractionColor (as the demo's glass any-hit does)
    const float3 o = hd->kind == DGB_MAT_GLASS
                         ? make_float3(1.0f - hd->base[0], 1.0f - hd->base[1], 1.0f - hd->base[2])
                         : surfaceOpacity(hd);
    const float a0 = __uint_as_float(optixGetPayload_0()) * (1.0f - o.x);
    const float a1 = __uint_as_float(optixGetPayload_1()) * (1.0f - o.y);
    const float a2 = __uint_as_float(optixGetPayload_2()) * (1.0f - o.z);
    optixSetPayload_0(__float_as_uint(a0));
    optixSetPayload_1(__float_as_uint(a1));
    optixSetPayload_2(__float_as_uint(a2));
    if (a0 + a1 + a2 < 0.004f) optixTerminateRay();
    optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__radiance()
{
    const HitData* hd = reinterpret_cast<const HitData*>(optixGetSbtDataPointer());
    const unsigned prim = optixGetPrimitiveIndex();
    const uint3 tri = hd->indices[prim];
    const float3 p0 = hd->positions[tri.x], p1 = hd->positions[tri.y], p2 = hd->positions[tri.z];
    const float2 b = optixGetTriangleBarycentrics();
    const float w0 = 1.0f - b.x - b.y;

    float3 ng = cross(p1 - p0, p2 - p0);
    float3 n = ng;
    if (hd->normals) n = hd->normals[tri.x] * w0 + hd->normals[tri.y] * b.x + hd->normals[tri.z] * b.y;

    // Bump map, as in the demo's phong shader (object space, texcoord1):
    // tangent-space normal = texel * 2 - 1 with z scaled by bumpScale; frame
    // from the interpolated normal and the tangent attribute (or any
    // perpendicular when the tangent is missing).
    if ((hd->componentState & CS_BUMP_TEX) && hd->bumpTex && hd->texcoords1 && hd->normals) {
        const float2 uv = interp2(hd->texcoords1, tri, b);
        const float4 tx = tex2D<float4>(hd->bumpTex, uv.x, uv.y);
        const float3 tn = normalize(make_float3(tx.x * 2.0f - 1.0f, tx.y * 2.0f - 1.0f,
                                                (tx.z * 2.0f - 1.0f) * hd->bumpScale));
        float3 bb;
        float3 t = make_float3(0.0f, 0.0f, 0.0f);
        if (hd->tangents) t = hd->tangents[tri.x] * w0 + hd->tangents[tri.y] * b.x + hd->tangents[tri.z] * b.y;
        const float ts = t.x + t.y + t.z;
        if (isfinite(ts) && ts != 0.0f) {
            bb = normalize(cross(n, t));
        } else {
            bb = fabsf(n.z) < fabsf(n.x) ? make_float3(-n.y, n.x, 0.0f) : make_float3(0.0f, -n.z, n.y);
            bb = normalize(bb);
        }
        const float3 tt = cross(bb, n);
        n = tt * tn.x + bb * tn.y + n * tn.z;
    }
    ng = normalize(optixTransformNormalFromObjectToWorldSpace(ng));
    const float3 nw = optixTransformNormalFromObjectToWorldSpace(n);
    const float len = length(nw);
    n = (len > 1e-12f && isfinite(len)) ? nw * (1.0f / len) : ng;

    const float3 dir = optixGetWorldRayDirection();
    if (dot(ng, dir) > 0.0f) ng = -ng;
    if (dot(n, ng) < 0.0f) n = -n;

    float3 base = f3(hd->base);
    if (hd->diffuseTex && hd->texcoords) {
        const float2 uv = interp2(hd->texcoords, tri, b);
        const float4 tc = tex2D<float4>(hd->diffuseTex, uv.x, uv.y);
        base = base * make_float3(tc.x * tc.x, tc.y * tc.y, tc.z * tc.z);   // diffuseColor * texture^2
    }

    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(__float_as_uint(n.x));
    optixSetPayload_2(__float_as_uint(n.y));
    optixSetPayload_3(__float_as_uint(n.z));
    optixSetPayload_4(__float_as_uint(ng.x));
    optixSetPayload_5(__float_as_uint(ng.y));
    optixSetPayload_6(__float_as_uint(ng.z));
    optixSetPayload_7(__float_as_uint(base.x));
    optixSetPayload_8(__float_as_uint(base.y));
    optixSetPayload_9(__float_as_uint(base.z));
    optixSetPayload_10(hd->materialIndex);
}
