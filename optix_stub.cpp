// optix_stub.cpp - DG Bridge, stage 4b: optix.1.dll replacement for Design Garage
//
// Replaces the OptiX 3.0 runtime used by NVIDIA Design Garage (32-bit).
// It implements the whole OptiX 3.0 C API, but instead of ray tracing it:
//   - keeps track of every object, variable and buffer the demo creates
//   - logs every call (with a per-function cap so per-frame calls don't flood)
//   - dumps every PTX program and (optionally) every input buffer's contents
//   - writes a snapshot of the scene graph whenever its structure changes
//   - stage 2: translates the scene for dgbridge_helper.exe (64-bit, OptiX 9),
//     which renders it; the image is copied into the demo's OpenGL buffer
//   - falls back to a moving test pattern if the helper is unavailable
//
// Output goes to %LOCALAPPDATA%\DGBridge\run_YYYYMMDD_HHMMSS\ (override with
// the DGBRIDGE_DIR environment variable).
//
// Environment switches (all optional):
//   DGBRIDGE_DIR=<folder>        where to write logs/dumps
//   DGBRIDGE_DUMP_BUFFERS=0|1    dump input buffer contents (default 0)
//   DGBRIDGE_TESTPATTERN=0|1     test pattern when the helper is unavailable (default 1)
//   DGBRIDGE_HELPER=0|1          use dgbridge_helper.exe to render (default 1)
//   DGBRIDGE_HELPER_CONSOLE=0|1  show the helper's console window (default 0)
//   DGBRIDGE_PHYSICAL=0|1        physically correct light balance instead of the
//                                original demo's (read by the helper; default 0)
//   DGBRIDGE_DENOISE=0|1         OptiX AI denoiser (read by the helper; default 0)
//   DGBRIDGE_SPP=<1..64>         light paths per pixel per frame (read by the helper; default 4)
//   DGBRIDGE_MAX_LOG=<n>         max logged calls per function (default 2000)
//
// Enum values and the version number were taken from the real OptiX 3.0.0
// DLLs that ship with Design Garage.

#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "dgbridge_protocol.h"

#define API extern "C"

typedef unsigned int RTresult;
typedef size_t RTsize;

// ---------------------------------------------------------------------------
// OptiX 3.0 constants
// ---------------------------------------------------------------------------

enum : unsigned {
    RT_SUCCESS = 0,
    RT_TIMEOUT_CALLBACK = 0x100,
    RT_ERROR_INVALID_CONTEXT = 0x500,
    RT_ERROR_INVALID_VALUE = 0x501,
    RT_ERROR_MEMORY_ALLOCATION_FAILED = 0x502,
    RT_ERROR_TYPE_MISMATCH = 0x503,
    RT_ERROR_VARIABLE_NOT_FOUND = 0x504,
    RT_ERROR_VARIABLE_REDECLARED = 0x505,
    RT_ERROR_ILLEGAL_SYMBOL = 0x506,
    RT_ERROR_INVALID_SOURCE = 0x507,
    RT_ERROR_VERSION_MISMATCH = 0x508,
    RT_ERROR_OBJECT_CREATION_FAILED = 0x600,
    RT_ERROR_NO_DEVICE = 0x601,
    RT_ERROR_INVALID_DEVICE = 0x602,
    RT_ERROR_INVALID_IMAGE = 0x603,
    RT_ERROR_FILE_NOT_FOUND = 0x604,
    RT_ERROR_ALREADY_MAPPED = 0x605,
    RT_ERROR_INVALID_DRIVER_VERSION = 0x606,
    RT_ERROR_CONTEXT_CREATION_FAILED = 0x607,
    RT_ERROR_RESOURCE_NOT_REGISTERED = 0x608,
    RT_ERROR_RESOURCE_ALREADY_REGISTERED = 0x609,
    RT_ERROR_LAUNCH_FAILED = 0x900,
    RT_ERROR_NOT_SUPPORTED = 0xA00,
    RT_ERROR_UNKNOWN = 0xFFFFFFFFu,
};

// RTobjecttype (verified against optixu.1.dll's rtuNameForType)
enum : unsigned {
    OT_UNKNOWN = 0x200, OT_GROUP, OT_GEOMETRY_GROUP, OT_TRANSFORM, OT_SELECTOR,
    OT_GEOMETRY_INSTANCE, OT_BUFFER, OT_TEXTURE_SAMPLER, OT_OBJECT,
    OT_MATRIX2x2, OT_MATRIX2x3, OT_MATRIX2x4, OT_MATRIX3x2, OT_MATRIX3x3,
    OT_MATRIX3x4, OT_MATRIX4x2, OT_MATRIX4x3, OT_MATRIX4x4,
    OT_FLOAT, OT_FLOAT2, OT_FLOAT3, OT_FLOAT4,
    OT_INT, OT_INT2, OT_INT3, OT_INT4,
    OT_UINT, OT_UINT2, OT_UINT3, OT_UINT4,
    OT_USER, OT_PROGRAM,
};

// RTformat (sizes verified against optixu.1.dll's rtuGetSizeForRTformat)
enum : unsigned { FMT_UNKNOWN = 0x100, FMT_FLOAT = 0x101, FMT_BYTE = 0x105,
                  FMT_UBYTE = 0x109, FMT_SHORT = 0x10D, FMT_USHORT = 0x111,
                  FMT_INT = 0x115, FMT_UINT = 0x119, FMT_USER = 0x11D };

enum : unsigned { BUF_INPUT = 0x1, BUF_OUTPUT = 0x2 };

static const char* resultName(RTresult r)
{
    switch (r) {
    case RT_SUCCESS: return "RT_SUCCESS";
    case RT_TIMEOUT_CALLBACK: return "RT_TIMEOUT_CALLBACK";
    case RT_ERROR_INVALID_CONTEXT: return "RT_ERROR_INVALID_CONTEXT";
    case RT_ERROR_INVALID_VALUE: return "RT_ERROR_INVALID_VALUE";
    case RT_ERROR_MEMORY_ALLOCATION_FAILED: return "RT_ERROR_MEMORY_ALLOCATION_FAILED";
    case RT_ERROR_TYPE_MISMATCH: return "RT_ERROR_TYPE_MISMATCH";
    case RT_ERROR_VARIABLE_NOT_FOUND: return "RT_ERROR_VARIABLE_NOT_FOUND";
    case RT_ERROR_VARIABLE_REDECLARED: return "RT_ERROR_VARIABLE_REDECLARED";
    case RT_ERROR_ILLEGAL_SYMBOL: return "RT_ERROR_ILLEGAL_SYMBOL";
    case RT_ERROR_INVALID_SOURCE: return "RT_ERROR_INVALID_SOURCE";
    case RT_ERROR_VERSION_MISMATCH: return "RT_ERROR_VERSION_MISMATCH";
    case RT_ERROR_OBJECT_CREATION_FAILED: return "RT_ERROR_OBJECT_CREATION_FAILED";
    case RT_ERROR_NO_DEVICE: return "RT_ERROR_NO_DEVICE";
    case RT_ERROR_INVALID_DEVICE: return "RT_ERROR_INVALID_DEVICE";
    case RT_ERROR_INVALID_IMAGE: return "RT_ERROR_INVALID_IMAGE";
    case RT_ERROR_FILE_NOT_FOUND: return "RT_ERROR_FILE_NOT_FOUND";
    case RT_ERROR_ALREADY_MAPPED: return "RT_ERROR_ALREADY_MAPPED";
    case RT_ERROR_INVALID_DRIVER_VERSION: return "RT_ERROR_INVALID_DRIVER_VERSION";
    case RT_ERROR_CONTEXT_CREATION_FAILED: return "RT_ERROR_CONTEXT_CREATION_FAILED";
    case RT_ERROR_RESOURCE_NOT_REGISTERED: return "RT_ERROR_RESOURCE_NOT_REGISTERED";
    case RT_ERROR_RESOURCE_ALREADY_REGISTERED: return "RT_ERROR_RESOURCE_ALREADY_REGISTERED";
    case RT_ERROR_LAUNCH_FAILED: return "RT_ERROR_LAUNCH_FAILED";
    case RT_ERROR_NOT_SUPPORTED: return "RT_ERROR_NOT_SUPPORTED";
    default: return "RT_ERROR_UNKNOWN";
    }
}

static const char* kTypeNames[] = {
    "unknown", "group", "geometrygroup", "transform", "selector", "geometryinstance",
    "buffer", "texturesampler", "object", "matrix2x2", "matrix2x3", "matrix2x4",
    "matrix3x2", "matrix3x3", "matrix3x4", "matrix4x2", "matrix4x3", "matrix4x4",
    "float", "float2", "float3", "float4", "int", "int2", "int3", "int4",
    "uint", "uint2", "uint3", "uint4", "userdata", "program" };

static const char* typeName(unsigned t)
{
    return (t >= OT_UNKNOWN && t <= OT_PROGRAM) ? kTypeNames[t - OT_UNKNOWN] : "?";
}

static std::string formatName(unsigned f)
{
    static const char* bases[] = { "FLOAT", "BYTE", "UNSIGNED_BYTE", "SHORT",
                                   "UNSIGNED_SHORT", "INT", "UNSIGNED_INT" };
    if (f == FMT_UNKNOWN) return "UNKNOWN";
    if (f == FMT_USER) return "USER";
    if (f > FMT_UNKNOWN && f < FMT_USER) {
        unsigned i = f - FMT_FLOAT;
        std::string s = bases[i / 4];
        if (i % 4) s += char('1' + i % 4);
        return s;
    }
    char b[32];
    std::snprintf(b, sizeof(b), "0x%x", f);
    return b;
}

// bytes per element for the standard formats, 0 for unknown/user
static size_t formatBytes(unsigned f)
{
    static const unsigned char comp[] = { 4, 1, 1, 2, 2, 4, 4 };
    if (f > FMT_UNKNOWN && f < FMT_USER) {
        unsigned i = f - FMT_FLOAT;
        return (size_t)comp[i / 4] * (i % 4 + 1);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------

enum Kind : unsigned { K_NONE, K_CONTEXT, K_PROGRAM, K_GROUP, K_SELECTOR, K_TRANSFORM,
                       K_GEOMGROUP, K_ACCEL, K_GINST, K_GEOMETRY, K_MATERIAL,
                       K_TEXSAMPLER, K_BUFFER, K_VARIABLE, K_COUNT };
static const char* kKindShort[K_COUNT] = { "?", "Ctx", "Prog", "Grp", "Sel", "Xf", "GGrp",
                                           "Accel", "GI", "Geom", "Mtl", "Tex", "Buf", "Var" };

struct Context;
struct Obj {
    Kind kind = K_NONE;
    unsigned id = 0;
    Context* ctx = nullptr;
    bool destroyed = false;
    virtual ~Obj() {}
};
struct Var;
struct Scope : Obj { std::vector<Var*> vars; };
struct Var : Obj {
    std::string name;
    Scope* owner = nullptr;
    unsigned type = OT_UNKNOWN;
    std::vector<unsigned char> data;
    Obj* object = nullptr;
};
struct Context : Scope {
    unsigned entryCount = 0, rayTypeCount = 0;
    std::map<unsigned, Obj*> raygen, exceptionProg, miss;
    std::map<unsigned, int> exceptionEnabled;
    RTsize stackSize = 1024;
    int printEnabled = 0;
    RTsize printBufferSize = 65536;
    int printIndex[3] = { -1, -1, -1 };
    std::vector<int> devices{ 0 };
    void* timeoutCallback = nullptr;
    double timeoutSeconds = 0;
    unsigned long long launches = 0;
    unsigned long long dumpedStructVersion = ~0ull;
};
struct Program : Scope { std::string name, file; unsigned long long ptxHash = 0; };
struct Group : Obj { std::vector<Obj*> children; Obj* accel = nullptr; };
struct Selector : Scope { std::vector<Obj*> children; Obj* visit = nullptr; };
struct Transform : Obj { float m[16], inv[16]; Obj* child = nullptr; };
struct GeomGroup : Obj { std::vector<Obj*> children; Obj* accel = nullptr; };
struct Accel : Obj {
    std::string builder = "NoAccel", traverser = "NoAccel";
    std::map<std::string, std::string> props;
    bool dirty = true;
};
struct GInst : Scope { Obj* geometry = nullptr; std::vector<Obj*> materials; };
struct Geometry : Scope { unsigned primCount = 0; Obj* bbox = nullptr; Obj* isect = nullptr; bool dirty = true; };
struct Material : Scope { std::map<unsigned, Obj*> ch, ah; };
struct TexSampler : Obj {
    unsigned mipCount = 1, arraySize = 1;
    unsigned wrap[3] = { 0, 0, 0 };
    unsigned filter[3] = { 1, 1, 2 };   // linear, linear, none
    float aniso = 1.0f;
    unsigned readMode = 0, indexMode = 0;
    std::map<std::pair<unsigned, unsigned>, Obj*> buffers;
    unsigned glId = 0, glTarget = 0;
};
struct Buffer : Obj {
    unsigned desc = 0, format = FMT_UNKNOWN;
    RTsize elemSize = 0;
    unsigned dimensionality = 1;
    RTsize dims[3] = { 0, 1, 1 };
    std::vector<unsigned char> host;
    bool mapped = false;
    unsigned glId = 0;
    unsigned long long lastHash = 0;
    size_t elementBytes() const {
        size_t f = formatBytes(format);
        return f ? f : elemSize;
    }
    size_t totalBytes() const { return (size_t)dims[0] * dims[1] * dims[2] * elementBytes(); }
};

static unsigned objectTypeOf(Obj* o)
{
    if (!o) return OT_UNKNOWN;
    switch (o->kind) {
    case K_GROUP: return OT_GROUP;
    case K_GEOMGROUP: return OT_GEOMETRY_GROUP;
    case K_TRANSFORM: return OT_TRANSFORM;
    case K_SELECTOR: return OT_SELECTOR;
    case K_GINST: return OT_GEOMETRY_INSTANCE;
    case K_BUFFER: return OT_BUFFER;
    case K_TEXSAMPLER: return OT_TEXTURE_SAMPLER;
    case K_PROGRAM: return OT_PROGRAM;
    default: return OT_OBJECT;
    }
}

// ---------------------------------------------------------------------------
// Global state, logging
// ---------------------------------------------------------------------------

static std::recursive_mutex g_mtx;
static bool g_inited = false;
static FILE* g_log = nullptr;
static std::string g_dir;
static unsigned long long g_seq = 0;
static std::map<const char*, unsigned long long> g_counts;
static unsigned long long g_maxLog = 2000;
static const char* g_lastCall = "(none)";
static bool g_dumpBuffers = false, g_testPattern = true;
static unsigned long long g_materialVersion = 0;
static HMODULE g_module = nullptr;
static unsigned long long g_dumpedBytes = 0;
static const unsigned long long g_dumpLimit = 3ull << 30;
static std::vector<Obj*> g_all;
static std::unordered_set<Obj*> g_valid;
static unsigned g_ids[K_COUNT];
static std::set<unsigned long long> g_ptxSeen;
static unsigned long long g_structVersion = 0;
static unsigned g_sceneDumps = 0;
static std::string g_lastError;
static LPTOP_LEVEL_EXCEPTION_FILTER g_prevFilter = nullptr;

static void logLine(const char* fmt, ...);

static LONG WINAPI crashFilter(EXCEPTION_POINTERS* ep)
{
    if (g_log) {
        std::fprintf(g_log, "\n!!! CRASH: exception 0x%08lx at address %p\n!!! last OptiX call: %s\n",
                     ep->ExceptionRecord->ExceptionCode, ep->ExceptionRecord->ExceptionAddress,
                     g_lastCall);
        std::fflush(g_log);
    }
    return g_prevFilter ? g_prevFilter(ep) : EXCEPTION_CONTINUE_SEARCH;
}

static bool envFlag(const char* name, bool def)
{
    char buf[16];
    DWORD n = GetEnvironmentVariableA(name, buf, sizeof(buf));
    if (n == 0 || n >= sizeof(buf)) return def;
    return buf[0] != '0';
}

static void ensureInit()
{
    if (g_inited) return;
    g_inited = true;

    char buf[MAX_PATH];
    DWORD n = GetEnvironmentVariableA("DGBRIDGE_DIR", buf, sizeof(buf));
    std::string root;
    if (n && n < sizeof(buf)) {
        root = buf;
    } else {
        n = GetEnvironmentVariableA("LOCALAPPDATA", buf, sizeof(buf));
        root = (n && n < sizeof(buf)) ? std::string(buf) + "\\DGBridge" : "C:\\DGBridge";
    }
    CreateDirectoryA(root.c_str(), nullptr);

    SYSTEMTIME st;
    GetLocalTime(&st);
    char run[64];
    std::snprintf(run, sizeof(run), "\\run_%04d%02d%02d_%02d%02d%02d",
                  st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    g_dir = root + run;
    CreateDirectoryA(g_dir.c_str(), nullptr);
    CreateDirectoryA((g_dir + "\\ptx").c_str(), nullptr);
    CreateDirectoryA((g_dir + "\\buffers").c_str(), nullptr);
    CreateDirectoryA((g_dir + "\\scenes").c_str(), nullptr);

    g_log = std::fopen((g_dir + "\\log.txt").c_str(), "w");

    g_dumpBuffers = envFlag("DGBRIDGE_DUMP_BUFFERS", false);
    g_testPattern = envFlag("DGBRIDGE_TESTPATTERN", true);
    n = GetEnvironmentVariableA("DGBRIDGE_MAX_LOG", buf, sizeof(buf));
    if (n && n < sizeof(buf)) g_maxLog = std::strtoull(buf, nullptr, 10);

    GetModuleFileNameA(nullptr, buf, sizeof(buf));
    logLine("DG Bridge (OptiX 3.0 API replacement) - stage 4b");
    logLine("process: %s", buf);
    logLine("output:  %s", g_dir.c_str());
    logLine("dump buffers: %d, test pattern: %d, max log per function: %llu",
            (int)g_dumpBuffers, (int)g_testPattern, g_maxLog);

    g_prevFilter = SetUnhandledExceptionFilter(crashFilter);
}

static void logLine(const char* fmt, ...)
{
    if (!g_log) return;
    std::fprintf(g_log, "[%8llu] ", g_seq);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_log, fmt, ap);
    va_end(ap);
    std::fputc('\n', g_log);
    std::fflush(g_log);
}

// Calls whose names contain these never change the scene structure.
static bool isStructural(const char* fn)
{
    return !(std::strstr(fn, "Get") || std::strstr(fn, "Query") || std::strstr(fn, "Validate") ||
             std::strstr(fn, "Launch") || std::strstr(fn, "Compile") || std::strstr(fn, "rtVariableSet") ||
             std::strstr(fn, "Map") || std::strstr(fn, "Unmap") || std::strstr(fn, "IsDirty") || std::strstr(fn, "rtDevice") ||
             std::strstr(fn, "MarkDirty"));
}

struct CallScope {
    bool log;
    explicit CallScope(const char* fn)
    {
        ensureInit();
        ++g_seq;
        g_lastCall = fn;
        unsigned long long c = ++g_counts[fn];
        log = c <= g_maxLog;
        if (c == g_maxLog + 1)
            logLine("(%s called more than %llu times; further successful calls are not logged)", fn, g_maxLog);
        if (isStructural(fn)) ++g_structVersion;
    }
};

#define ENTER(fn)                                         \
    std::lock_guard<std::recursive_mutex> _lk(g_mtx);     \
    CallScope _cs(fn)

static RTresult R(const CallScope& cs, RTresult r, const char* fmt, ...)
{
    if (cs.log || r != RT_SUCCESS) {
        char buf[2048];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        logLine("%s -> %s", buf, resultName(r));
        if (r != RT_SUCCESS) g_lastError = buf;
    }
    return r;
}

// Short printable name for an object handle, e.g. "Buf#12".
static const char* N(const void* h)
{
    static char ring[32][48];
    static unsigned idx = 0;
    char* s = ring[idx++ % 32];
    const Obj* o = (const Obj*)h;
    if (!h) std::snprintf(s, 48, "NULL");
    else if (!g_valid.count((Obj*)o)) std::snprintf(s, 48, "<invalid %p>", h);
    else std::snprintf(s, 48, "%s#%u%s", kKindShort[o->kind], o->id, o->destroyed ? "(destroyed)" : "");
    return s;
}

template <class T>
static T* mk(Kind k, Context* ctx)
{
    T* o = new T();
    o->kind = k;
    o->id = ++g_ids[k];
    o->ctx = ctx;
    g_all.push_back(o);
    g_valid.insert(o);
    return o;
}

template <class T>
static T* as(void* h, Kind k)
{
    Obj* o = (Obj*)h;
    if (!o || !g_valid.count(o) || o->kind != k || o->destroyed) return nullptr;
    return static_cast<T*>(o);
}

static Obj* asAny(void* h)
{
    Obj* o = (Obj*)h;
    if (!o || !g_valid.count(o) || o->destroyed) return nullptr;
    return o;
}

static RTresult badHandle(const CallScope& cs, const char* fn, Kind k, const void* h)
{
    return R(cs, k == K_CONTEXT ? RT_ERROR_INVALID_CONTEXT : RT_ERROR_INVALID_VALUE,
             "%s(%s) [bad handle, expected %s]", fn, N(h), kKindShort[k]);
}

static unsigned long long fnv1a(const void* p, size_t n)
{
    const unsigned char* b = (const unsigned char*)p;
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static bool writeFile(const std::string& path, const void* data, size_t n)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(data, 1, n, f) == n;
    std::fclose(f);
    return ok;
}

// ---------------------------------------------------------------------------
// Value formatting for the logs
// ---------------------------------------------------------------------------

static std::string fmtFloats(const float* v, int n)
{
    std::string s = "(";
    char b[32];
    for (int i = 0; i < n; ++i) {
        std::snprintf(b, sizeof(b), "%s%g", i ? ", " : "", v[i]);
        s += b;
    }
    return s + ")";
}

static std::string fmtInts(const int* v, int n)
{
    std::string s = "(";
    char b[32];
    for (int i = 0; i < n; ++i) {
        std::snprintf(b, sizeof(b), "%s%d", i ? ", " : "", v[i]);
        s += b;
    }
    return s + ")";
}

static std::string fmtUints(const unsigned* v, int n)
{
    std::string s = "(";
    char b[32];
    for (int i = 0; i < n; ++i) {
        std::snprintf(b, sizeof(b), "%s%u", i ? ", " : "", v[i]);
        s += b;
    }
    return s + ")";
}

static std::string fmtHex(const void* p, size_t n, size_t maxBytes = 32)
{
    std::string s;
    char b[4];
    const unsigned char* c = (const unsigned char*)p;
    for (size_t i = 0; i < n && i < maxBytes; ++i) {
        std::snprintf(b, sizeof(b), "%02x", c[i]);
        s += b;
    }
    if (n > maxBytes) s += "...";
    return s;
}

static std::string fmtVar(const Var* v)
{
    const unsigned t = v->type;
    if (v->object || (t >= OT_GROUP && t <= OT_OBJECT) || t == OT_PROGRAM)
        return std::string(typeName(t)) + " " + N(v->object);
    const size_t n = v->data.size();
    if (t >= OT_FLOAT && t <= OT_FLOAT4)
        return std::string(typeName(t)) + fmtFloats((const float*)v->data.data(), (int)(n / 4));
    if (t >= OT_MATRIX2x2 && t <= OT_MATRIX4x4)
        return std::string(typeName(t)) + fmtFloats((const float*)v->data.data(), (int)(n / 4));
    if (t >= OT_INT && t <= OT_INT4)
        return std::string(typeName(t)) + fmtInts((const int*)v->data.data(), (int)(n / 4));
    if (t >= OT_UINT && t <= OT_UINT4)
        return std::string(typeName(t)) + fmtUints((const unsigned*)v->data.data(), (int)(n / 4));
    if (t == OT_USER) {
        char b[32];
        std::snprintf(b, sizeof(b), "userdata[%u] ", (unsigned)n);
        return b + fmtHex(v->data.data(), n, 64);
    }
    return "(unset)";
}

// ---------------------------------------------------------------------------
// Scene snapshots
// ---------------------------------------------------------------------------

static void printVars(FILE* f, const Scope* s, const char* indent)
{
    for (const Var* v : s->vars)
        if (!v->destroyed)
            std::fprintf(f, "%s  var %-28s = %s\n", indent, v->name.c_str(), fmtVar(v).c_str());
}

static void describe(FILE* f, Obj* o)
{
    std::fprintf(f, "%s", N(o));
    switch (o->kind) {
    case K_CONTEXT: {
        Context* c = (Context*)o;
        std::fprintf(f, "  entries=%u rayTypes=%u stack=%u launches=%llu\n", c->entryCount,
                     c->rayTypeCount, (unsigned)c->stackSize, c->launches);
        for (auto& e : c->raygen) std::fprintf(f, "  raygen[%u] = %s\n", e.first, N(e.second));
        for (auto& e : c->exceptionProg) std::fprintf(f, "  exception[%u] = %s\n", e.first, N(e.second));
        for (auto& e : c->miss) std::fprintf(f, "  miss[%u] = %s\n", e.first, N(e.second));
        for (auto& e : c->exceptionEnabled) std::fprintf(f, "  exceptionEnabled[0x%x] = %d\n", e.first, e.second);
        printVars(f, c, "");
        break;
    }
    case K_PROGRAM: {
        Program* p = (Program*)o;
        std::fprintf(f, "  \"%s\" ptx=%016llx%s%s\n", p->name.c_str(), p->ptxHash,
                     p->file.empty() ? "" : " file=", p->file.c_str());
        printVars(f, p, "");
        break;
    }
    case K_GROUP: {
        Group* g = (Group*)o;
        std::fprintf(f, "  accel=%s children=%u:", N(g->accel), (unsigned)g->children.size());
        for (Obj* c : g->children) std::fprintf(f, " %s", N(c));
        std::fprintf(f, "\n");
        break;
    }
    case K_SELECTOR: {
        Selector* s = (Selector*)o;
        std::fprintf(f, "  visit=%s children=%u:", N(s->visit), (unsigned)s->children.size());
        for (Obj* c : s->children) std::fprintf(f, " %s", N(c));
        std::fprintf(f, "\n");
        printVars(f, s, "");
        break;
    }
    case K_TRANSFORM: {
        Transform* t = (Transform*)o;
        std::fprintf(f, "  child=%s matrix=%s\n", N(t->child), fmtFloats(t->m, 16).c_str());
        break;
    }
    case K_GEOMGROUP: {
        GeomGroup* g = (GeomGroup*)o;
        std::fprintf(f, "  accel=%s children=%u:", N(g->accel), (unsigned)g->children.size());
        for (Obj* c : g->children) std::fprintf(f, " %s", N(c));
        std::fprintf(f, "\n");
        break;
    }
    case K_ACCEL: {
        Accel* a = (Accel*)o;
        std::fprintf(f, "  builder=%s traverser=%s dirty=%d", a->builder.c_str(),
                     a->traverser.c_str(), (int)a->dirty);
        for (auto& p : a->props) std::fprintf(f, " %s=%s", p.first.c_str(), p.second.c_str());
        std::fprintf(f, "\n");
        break;
    }
    case K_GINST: {
        GInst* g = (GInst*)o;
        std::fprintf(f, "  geometry=%s materials:", N(g->geometry));
        for (Obj* m : g->materials) std::fprintf(f, " %s", N(m));
        std::fprintf(f, "\n");
        printVars(f, g, "");
        break;
    }
    case K_GEOMETRY: {
        Geometry* g = (Geometry*)o;
        std::fprintf(f, "  prims=%u bbox=%s isect=%s\n", g->primCount, N(g->bbox), N(g->isect));
        printVars(f, g, "");
        break;
    }
    case K_MATERIAL: {
        Material* m = (Material*)o;
        std::fprintf(f, " ");
        for (auto& e : m->ch) std::fprintf(f, " ch[%u]=%s", e.first, N(e.second));
        for (auto& e : m->ah) std::fprintf(f, " ah[%u]=%s", e.first, N(e.second));
        std::fprintf(f, "\n");
        printVars(f, m, "");
        break;
    }
    case K_TEXSAMPLER: {
        TexSampler* t = (TexSampler*)o;
        std::fprintf(f, "  wrap=(%u,%u,%u) filter=(%u,%u,%u) aniso=%g read=%u index=%u mips=%u array=%u gl=%u",
                     t->wrap[0], t->wrap[1], t->wrap[2], t->filter[0], t->filter[1], t->filter[2],
                     t->aniso, t->readMode, t->indexMode, t->mipCount, t->arraySize, t->glId);
        for (auto& b : t->buffers) std::fprintf(f, " buf[%u,%u]=%s", b.first.first, b.first.second, N(b.second));
        std::fprintf(f, "\n");
        break;
    }
    case K_BUFFER: {
        Buffer* b = (Buffer*)o;
        std::fprintf(f, "  desc=0x%x %s elem=%u dims=%ux%ux%u (%uD) gl=%u hash=%016llx\n", b->desc,
                     formatName(b->format).c_str(), (unsigned)b->elementBytes(), (unsigned)b->dims[0],
                     (unsigned)b->dims[1], (unsigned)b->dims[2], b->dimensionality, b->glId, b->lastHash);
        break;
    }
    default:
        std::fprintf(f, "\n");
    }
}

static void dumpScene(Context* c, const char* reason)
{
    if (g_sceneDumps >= 200) return;
    char name[64];
    std::snprintf(name, sizeof(name), "\\scenes\\scene_%03u.txt", g_sceneDumps++);
    FILE* f = std::fopen((g_dir + name).c_str(), "w");
    if (!f) return;
    std::fprintf(f, "Scene snapshot (%s) at call #%llu\n\n", reason, g_seq);
    for (unsigned k = K_CONTEXT; k < K_VARIABLE; ++k) {
        bool header = false;
        for (Obj* o : g_all) {
            if (o->kind != (Kind)k || o->destroyed || (c && o->ctx != c)) continue;
            if (!header) {
                std::fprintf(f, "==== %s ====\n", kKindShort[k]);
                header = true;
            }
            describe(f, o);
        }
        if (header) std::fprintf(f, "\n");
    }
    std::fclose(f);
    logLine("wrote scene snapshot %s (%s)", name + 1, reason);
}

// ---------------------------------------------------------------------------
// Buffer dumps and the GL test pattern
// ---------------------------------------------------------------------------

static void maybeDumpBuffer(Buffer* b)
{
    if (b->host.empty()) return;
    const size_t n = std::min(b->host.size(), b->totalBytes() ? b->totalBytes() : b->host.size());
    const unsigned long long h = fnv1a(b->host.data(), n);
    if (h == b->lastHash) return;
    b->lastHash = h;
    logLine("  %s contents: %u bytes, hash %016llx, first bytes %s", N(b), (unsigned)n, h,
            fmtHex(b->host.data(), n, 48).c_str());
    if (!g_dumpBuffers || !(b->desc & BUF_INPUT) || g_dumpedBytes + n > g_dumpLimit) return;
    char name[160];
    std::snprintf(name, sizeof(name), "\\buffers\\buf%04u_%s_%ux%ux%u_%016llx.bin", b->id,
                  formatName(b->format).c_str(), (unsigned)b->dims[0], (unsigned)b->dims[1],
                  (unsigned)b->dims[2], h);
    if (writeFile(g_dir + name, b->host.data(), n)) g_dumpedBytes += n;
}

typedef void(APIENTRY* PFN_glBindBuffer)(unsigned, unsigned);
typedef void(APIENTRY* PFN_glBufferSubData)(unsigned, ptrdiff_t, ptrdiff_t, const void*);
typedef void(APIENTRY* PFN_glGetBufferParameteriv)(unsigned, unsigned, int*);
typedef void(APIENTRY* PFN_glGetIntegerv)(unsigned, int*);
typedef PROC(WINAPI* PFN_wglGetProcAddress)(LPCSTR);
typedef HGLRC(WINAPI* PFN_wglGetCurrentContext)(void);

static const unsigned GL_ARRAY_BUFFER_ = 0x8892, GL_ARRAY_BUFFER_BINDING_ = 0x8894, GL_BUFFER_SIZE_ = 0x8764;

static struct {
    bool ok = false, loggedWait = false;
    unsigned attempts = 0;
    PFN_glBindBuffer bindBuffer;
    PFN_glBufferSubData bufferSubData;
    PFN_glGetBufferParameteriv getBufferParameteriv;
    PFN_glGetIntegerv getIntegerv;
    PFN_wglGetCurrentContext getCurrentContext;
} g_gl;

// Resolves the few OpenGL functions we need. OpenGL may not be loaded or
// current during the first launches, so this keeps retrying for a while.
static bool glReady()
{
    if (!g_gl.ok) {
        if (g_gl.attempts > 1000) return false;
        if (++g_gl.attempts == 1001) {
            logLine("OpenGL: gave up - no usable OpenGL context on the launching thread");
            return false;
        }
        HMODULE gl = GetModuleHandleA("opengl32.dll");
        if (gl) {
            auto gpa = (PFN_wglGetProcAddress)(void*)GetProcAddress(gl, "wglGetProcAddress");
            g_gl.getCurrentContext = (PFN_wglGetCurrentContext)(void*)GetProcAddress(gl, "wglGetCurrentContext");
            g_gl.getIntegerv = (PFN_glGetIntegerv)(void*)GetProcAddress(gl, "glGetIntegerv");
            if (gpa && g_gl.getCurrentContext && g_gl.getCurrentContext()) {
                g_gl.bindBuffer = (PFN_glBindBuffer)(void*)gpa("glBindBuffer");
                g_gl.bufferSubData = (PFN_glBufferSubData)(void*)gpa("glBufferSubData");
                g_gl.getBufferParameteriv = (PFN_glGetBufferParameteriv)(void*)gpa("glGetBufferParameteriv");
            }
        }
        g_gl.ok = g_gl.bindBuffer && g_gl.bufferSubData && g_gl.getBufferParameteriv && g_gl.getIntegerv;
        if (g_gl.ok) {
            logLine("OpenGL: functions found");
        } else if (!g_gl.loggedWait) {
            g_gl.loggedWait = true;
            logLine("OpenGL: no context yet, will keep trying");
        }
        if (!g_gl.ok) return false;
    }
    return g_gl.getCurrentContext() != nullptr;
}

// Binds a GL buffer object; returns its size and the previous binding.
static int glBindForUpload(unsigned glId, int* prevBinding)
{
    int size = 0;
    g_gl.getIntegerv(GL_ARRAY_BUFFER_BINDING_, prevBinding);
    g_gl.bindBuffer(GL_ARRAY_BUFFER_, glId);
    g_gl.getBufferParameteriv(GL_ARRAY_BUFFER_, GL_BUFFER_SIZE_, &size);
    return size;
}

static void fillTestPattern(Context* c)
{
    if (!glReady()) return;
    const float t = (float)(c->launches % 360) * (3.14159265f / 180.0f);
    static std::vector<unsigned char> tmp;

    for (Obj* o : g_all) {
        if (o->kind != K_BUFFER || o->destroyed || o->ctx != c) continue;
        Buffer* b = (Buffer*)o;
        if (!b->glId || !(b->desc & BUF_OUTPUT)) continue;
        const unsigned w = (unsigned)b->dims[0], h = (unsigned)b->dims[1];
        const size_t eb = b->elementBytes();
        if (!w || !eb) continue;

        int prev = 0;
        const int glSize = glBindForUpload(b->glId, &prev);
        const size_t bytes = std::min((size_t)glSize, b->totalBytes());
        tmp.resize(bytes);

        const unsigned f = b->format;
        const bool isFloat = f >= FMT_FLOAT && f <= FMT_FLOAT + 3;
        const bool isByte = f >= FMT_UBYTE && f <= FMT_UBYTE + 3;
        const unsigned comps = (f > FMT_UNKNOWN && f < FMT_USER) ? (f - FMT_FLOAT) % 4 + 1 : 0;
        if ((isFloat || isByte) && comps) {
            for (unsigned y = 0; y < h; ++y)
                for (unsigned x = 0; x < w; ++x) {
                    const size_t off = ((size_t)y * w + x) * eb;
                    if (off + eb > bytes) break;
                    float rgba[4] = { (float)x / w, (float)y / h, 0.5f + 0.5f * std::sin(t + x * 0.02f), 1.0f };
                    if (((x / 64) + (y / 64)) % 2) { rgba[0] *= 0.6f; rgba[1] *= 0.6f; }
                    for (unsigned k = 0; k < comps; ++k) {
                        if (isFloat) std::memcpy(&tmp[off + 4 * k], &rgba[k], 4);
                        else tmp[off + k] = (unsigned char)(rgba[k] * 255.0f);
                    }
                }
            g_gl.bufferSubData(GL_ARRAY_BUFFER_, 0, (ptrdiff_t)bytes, tmp.data());
        }
        g_gl.bindBuffer(GL_ARRAY_BUFFER_, (unsigned)prev);
    }
}

// ===========================================================================
// Bridge to the 64-bit render helper (stage 2)
// ===========================================================================

static struct {
    bool tried = false, ok = false;
    HANDLE map = nullptr, req = nullptr, resp = nullptr, proc = nullptr;
    DgbHeader* hdr = nullptr;
    unsigned char* data = nullptr;
    size_t fill = 0;
    unsigned long long syncedStruct = ~0ull, syncedMaterial = ~0ull;
    std::set<unsigned long long> sentMeshes, sentTextures;
    unsigned errors = 0;
    unsigned long long frames = 0;
} g_br;

static void bridgeShutdown(const char* why)
{
    logLine("bridge: disabled (%s); falling back to the test pattern", why);
    g_br.ok = false;
}

// Sends the command in the header and waits for the helper's answer.
static bool bridgeCall(uint32_t cmd, DWORD timeoutMs)
{
    if (!g_br.ok) return false;
    g_br.hdr->cmd = cmd;
    g_br.hdr->status = 0xFFFFFFFFu;
    g_br.hdr->msg[0] = 0;
    SetEvent(g_br.req);
    HANDLE waits[2] = { g_br.resp, g_br.proc };
    const DWORD r = WaitForMultipleObjects(2, waits, FALSE, timeoutMs);
    if (r == WAIT_OBJECT_0) {
        if (g_br.hdr->status == 0) return true;
        if (++g_br.errors <= 50) logLine("bridge: helper error on command %u: %s", cmd, g_br.hdr->msg);
        return false;
    }
    bridgeShutdown(r == WAIT_OBJECT_0 + 1 ? "helper process exited - see helper_log.txt" : "helper timed out");
    return false;
}

static void bridgeStart()
{
    g_br.tried = true;
    if (!envFlag("DGBRIDGE_HELPER", true)) {
        logLine("bridge: helper disabled by DGBRIDGE_HELPER=0");
        return;
    }
    char path[MAX_PATH];
    GetModuleFileNameA(g_module, path, MAX_PATH);
    std::string dir = path;
    dir = dir.substr(0, dir.find_last_of("\\/"));
    const std::string exe = dir + "\\dgbridge_helper.exe";
    if (GetFileAttributesA(exe.c_str()) == INVALID_FILE_ATTRIBUTES) {
        logLine("bridge: %s not found - test pattern only", exe.c_str());
        return;
    }

    const DWORD pid = GetCurrentProcessId();
    char name[128];
    std::snprintf(name, sizeof(name), "Local\\DGBridge_%lu_shm", pid);
    g_br.map = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                  DGB_SHM_HEADER_SIZE + DGB_SHM_DATA_SIZE, name);
    void* view = g_br.map ? MapViewOfFile(g_br.map, FILE_MAP_ALL_ACCESS, 0, 0, 0) : nullptr;
    std::snprintf(name, sizeof(name), "Local\\DGBridge_%lu_req", pid);
    g_br.req = CreateEventA(nullptr, FALSE, FALSE, name);
    std::snprintf(name, sizeof(name), "Local\\DGBridge_%lu_resp", pid);
    g_br.resp = CreateEventA(nullptr, FALSE, FALSE, name);
    if (!view || !g_br.req || !g_br.resp) {
        logLine("bridge: could not create shared memory/events (error %lu)", GetLastError());
        return;
    }
    g_br.hdr = (DgbHeader*)view;
    g_br.data = (unsigned char*)view + DGB_SHM_HEADER_SIZE;
    std::memset(g_br.hdr, 0, sizeof(DgbHeader));
    g_br.hdr->magic = DGB_MAGIC;
    g_br.hdr->version = DGB_PROTOCOL_VERSION;

    std::string cmd = "\"" + exe + "\" " + std::to_string(pid) + " \"" + g_dir + "\"";
    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};
    const DWORD flags = envFlag("DGBRIDGE_HELPER_CONSOLE", false) ? CREATE_NEW_CONSOLE : CREATE_NO_WINDOW;
    if (!CreateProcessA(nullptr, &cmd[0], nullptr, nullptr, FALSE, flags, nullptr, dir.c_str(), &si, &pi)) {
        logLine("bridge: could not start helper (error %lu)", GetLastError());
        return;
    }
    CloseHandle(pi.hThread);
    g_br.proc = pi.hProcess;
    g_br.ok = true;
    logLine("bridge: started helper (pid %lu), initializing...", pi.dwProcessId);
    if (!bridgeCall(DGB_HELLO, 120000)) {
        bridgeShutdown("helper failed to initialize - see helper_log.txt");
        return;
    }
    logLine("bridge: helper ready: %s", g_br.hdr->msg);
}

// ---- scene stream writer ----

static bool flushChunk()
{
    if (!g_br.fill) return g_br.ok;
    g_br.hdr->dataSize = g_br.fill;
    const bool ok = bridgeCall(DGB_CHUNK, 60000);
    g_br.fill = 0;
    return ok;
}

static bool putBytes(const void* p, size_t n)
{
    const unsigned char* s = (const unsigned char*)p;
    while (n) {
        const size_t k = std::min(n, (size_t)DGB_SHM_DATA_SIZE - g_br.fill);
        std::memcpy(g_br.data + g_br.fill, s, k);
        g_br.fill += k;
        s += k;
        n -= k;
        if (g_br.fill == DGB_SHM_DATA_SIZE && !flushChunk()) return false;
    }
    return true;
}

// ---- reading the demo's scene ----

static Var* findVar(const Scope* s, const char* name)
{
    if (!s) return nullptr;
    for (Var* v : s->vars)
        if (!v->destroyed && v->name == name) return v;
    return nullptr;
}

// Copies up to n floats from a float/float2/3/4 variable; returns how many.
static int varFloats(const Var* v, float* out, int n)
{
    if (!v || v->type < OT_FLOAT || v->type > OT_FLOAT4) return 0;
    const int have = std::min(n, (int)(v->data.size() / 4));
    std::memcpy(out, v->data.data(), have * 4);
    return have;
}

static bool varUints(const Var* v, unsigned* out, int n)
{
    if (!v || v->data.size() < (size_t)n * 4) return false;
    std::memcpy(out, v->data.data(), n * 4);
    return true;
}

static Buffer* varBuffer(const Scope* s, const char* name)
{
    Var* v = findVar(s, name);
    return (v && v->object && v->object->kind == K_BUFFER && !v->object->destroyed) ? (Buffer*)v->object : nullptr;
}

static void mul4x4(const float* a, const float* b, float* o)
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            o[r * 4 + c] = a[r * 4 + 0] * b[0 * 4 + c] + a[r * 4 + 1] * b[1 * 4 + c] + a[r * 4 + 2] * b[2 * 4 + c] +
                           a[r * 4 + 3] * b[3 * 4 + c];
}

struct SceneInst {
    Geometry* geo;
    Material* mat;
    float M[16];
};

static void addInstance(GInst* gi, const float* M, std::vector<SceneInst>& out)
{
    if (!gi || gi->destroyed || !gi->geometry || gi->geometry->destroyed) return;
    SceneInst si;
    si.geo = (Geometry*)gi->geometry;
    si.mat = (!gi->materials.empty() && gi->materials[0] && !gi->materials[0]->destroyed) ? (Material*)gi->materials[0]
                                                                                            : nullptr;
    std::memcpy(si.M, M, sizeof(si.M));
    out.push_back(si);
}

static void collect(Obj* n, const float* M, std::vector<SceneInst>& out, int depth)
{
    if (!n || n->destroyed || depth > 32) return;
    switch (n->kind) {
    case K_GROUP:
        for (Obj* c : ((Group*)n)->children) collect(c, M, out, depth + 1);
        break;
    case K_SELECTOR:   // not used by Design Garage; take the first child
        for (Obj* c : ((Selector*)n)->children)
            if (c) { collect(c, M, out, depth + 1); break; }
        break;
    case K_TRANSFORM: {
        float M2[16];
        mul4x4(M, ((Transform*)n)->m, M2);
        collect(((Transform*)n)->child, M2, out, depth + 1);
        break;
    }
    case K_GEOMGROUP:
        for (Obj* c : ((GeomGroup*)n)->children) addInstance((GInst*)c, M, out);
        break;
    case K_GINST:
        addInstance((GInst*)n, M, out);
        break;
    default:
        break;
    }
}

static Obj* findTopObject(Context* c)
{
    static const char* names[] = { "topObject", "top_object" };
    for (const char* nm : names) {
        Var* v = findVar(c, nm);
        if (v && v->object) return v->object;
    }
    for (Var* v : c->vars)
        if (!v->destroyed && v->object &&
            (v->object->kind == K_GROUP || v->object->kind == K_GEOMGROUP || v->object->kind == K_TRANSFORM))
            return v->object;
    return nullptr;
}

struct MeshSrc {
    Buffer *ib = nullptr, *vb = nullptr, *ab = nullptr;
    unsigned vstride = 0, astride = 0, nv = 0, nt = 0;
    unsigned nrmSlot = 0, tcSlot[2] = { 0, 0 }, tanSlot = 0;
    uint32_t flags = 0;
    unsigned long long key = 0;
};

static bool meshSource(Geometry* g, MeshSrc& s)
{
    s.ib = varBuffer(g, "index_buffer");
    s.vb = varBuffer(g, "vertex_buffer");
    if (!s.ib || !s.vb || s.ib->host.empty() || s.vb->host.empty()) return false;
    s.vstride = (unsigned)s.vb->elementBytes();
    if (s.vstride != 12 && s.vstride != 16) return false;
    if (s.ib->elementBytes() != 12) return false;
    s.nv = (unsigned)s.vb->dims[0];
    s.nt = std::min(g->primCount, (unsigned)s.ib->dims[0]);
    if (!s.nv || !s.nt) return false;
    if (s.vb->host.size() < (size_t)s.nv * s.vstride || s.ib->host.size() < (size_t)s.nt * 12) return false;

    // Per-vertex attributes: SceniX stores them in attribute_buffer,
    // vertex_stride bytes per vertex. attribute_offsets (uint2) packs one 4-bit
    // float4 slot index per attribute, in SceniX's own intersection program's
    // order: x nibble 2 = normal, x nibble 7 = tangent, y nibble 0 = texcoord0,
    // y nibble 1 = texcoord1.
    s.ab = varBuffer(g, "attribute_buffer");
    unsigned offs[2] = { 0, 0 }, stride = 0;
    if (s.ab && !s.ab->host.empty() && varUints(findVar(g, "attribute_offsets"), offs, 2) &&
        varUints(findVar(g, "vertex_stride"), &stride, 1) && stride >= 16) {
        s.astride = stride;
        s.nrmSlot = (offs[0] >> 8) & 0xF;
        s.tcSlot[0] = offs[1] & 0xF;
        s.tcSlot[1] = (offs[1] >> 4) & 0xF;
        s.tanSlot = (offs[0] >> 28) & 0xF;
        auto fits = [&](unsigned slot, unsigned bytes) {
            return (size_t)(s.nv - 1) * stride + slot * 16 + bytes <= s.ab->host.size() && slot * 16 + bytes <= stride;
        };
        if (fits(s.nrmSlot, 12)) s.flags |= DGB_MESH_NORMALS;
        if (fits(s.tcSlot[0], 8)) s.flags |= DGB_MESH_TEXCOORD0;
        if (fits(s.tcSlot[1], 8)) s.flags |= DGB_MESH_TEXCOORD1;
        if (fits(s.tanSlot, 12)) s.flags |= DGB_MESH_TANGENTS;
    }
    const unsigned long long parts[12] = { s.ib->lastHash, s.vb->lastHash, s.flags ? s.ab->lastHash : 0,
                                           s.nv, s.nt, s.nrmSlot, s.tcSlot[0], s.tcSlot[1], s.tanSlot, s.astride,
                                           s.flags, s.vstride };
    s.key = fnv1a(parts, sizeof(parts));
    return true;
}

static bool writeMesh(const MeshSrc& s)
{
    const DgbRecord rec = { DGB_REC_MESH, 0 };
    const DgbMeshHeader mh = { s.key, s.nv, s.nt, s.flags, 0u };
    if (!putBytes(&rec, sizeof(rec)) || !putBytes(&mh, sizeof(mh))) return false;

    float tmp[3 * 1024];
    unsigned n = 0;
    for (unsigned i = 0; i < s.nv; ++i) {
        std::memcpy(&tmp[3 * n], s.vb->host.data() + (size_t)i * s.vstride, 12);
        if (++n == 1024) { if (!putBytes(tmp, sizeof(tmp))) return false; n = 0; }
    }
    if (n && !putBytes(tmp, n * 12)) return false;
    n = 0;
    if (s.flags & DGB_MESH_NORMALS) {
        for (unsigned i = 0; i < s.nv; ++i) {
            std::memcpy(&tmp[3 * n], s.ab->host.data() + (size_t)i * s.astride + s.nrmSlot * 16, 12);
            if (++n == 1024) { if (!putBytes(tmp, sizeof(tmp))) return false; n = 0; }
        }
        if (n && !putBytes(tmp, n * 12)) return false;
        n = 0;
    }
    for (int set = 0; set < 2; ++set) {
        if (!(s.flags & (set ? DGB_MESH_TEXCOORD1 : DGB_MESH_TEXCOORD0))) continue;
        for (unsigned i = 0; i < s.nv; ++i) {
            std::memcpy(&tmp[2 * n], s.ab->host.data() + (size_t)i * s.astride + s.tcSlot[set] * 16, 8);
            if (++n == 1536) { if (!putBytes(tmp, 1536 * 8)) return false; n = 0; }
        }
        if (n && !putBytes(tmp, n * 8)) return false;
        n = 0;
    }
    if (s.flags & DGB_MESH_TANGENTS) {
        for (unsigned i = 0; i < s.nv; ++i) {
            std::memcpy(&tmp[3 * n], s.ab->host.data() + (size_t)i * s.astride + s.tanSlot * 16, 12);
            if (++n == 1024) { if (!putBytes(tmp, sizeof(tmp))) return false; n = 0; }
        }
        if (n && !putBytes(tmp, n * 12)) return false;
        n = 0;
    }
    unsigned idx[3 * 1024];
    for (unsigned t = 0; t < s.nt; ++t) {
        std::memcpy(&idx[3 * n], s.ib->host.data() + (size_t)t * 12, 12);
        for (int k = 0; k < 3; ++k)
            if (idx[3 * n + k] >= s.nv) idx[3 * n + k] = 0;   // keep the GPU safe from bad indices
        if (++n == 1024) { if (!putBytes(idx, sizeof(idx))) return false; n = 0; }
    }
    return !n || putBytes(idx, n * 12);
}

// Texture image behind a sampler variable, if it is one we can send (RGBA8, 2D).
static Buffer* textureBuffer(const Var* v, TexSampler** sampler)
{
    if (!v || !v->object || v->object->kind != K_TEXSAMPLER || v->object->destroyed) return nullptr;
    TexSampler* t = (TexSampler*)v->object;
    auto it = t->buffers.find({ 0u, 0u });
    if (it == t->buffers.end() || !it->second || it->second->kind != K_BUFFER || it->second->destroyed) return nullptr;
    Buffer* b = (Buffer*)it->second;
    if (b->format != FMT_UBYTE + 3 || b->dims[0] < 1 || b->dims[1] < 1 || b->dims[2] != 1) return nullptr;
    if (b->host.size() < (size_t)b->dims[0] * b->dims[1] * 4) return nullptr;
    if (sampler) *sampler = t;
    return b;
}

static unsigned long long textureKey(const Buffer* b)
{
    const unsigned long long parts[3] = { b->lastHash, b->dims[0], b->dims[1] };
    return fnv1a(parts, sizeof(parts));
}

static bool writeTexture(const Buffer* b, unsigned long long key)
{
    const DgbRecord rec = { DGB_REC_TEXTURE, 0 };
    const DgbTextureHeader th = { key, (uint32_t)b->dims[0], (uint32_t)b->dims[1] };
    return putBytes(&rec, sizeof(rec)) && putBytes(&th, sizeof(th)) &&
           putBytes(b->host.data(), (size_t)b->dims[0] * b->dims[1] * 4);
}

// A texture a phong material actually uses, as in the original phong shader:
// componentState bit 'bit' switches it on (4 diffuse, 16 transparency, 1024 bump).
static Buffer* phongTexture(const Material* m, const char* var, unsigned bit, const char* setVar, uint32_t* sampler)
{
    unsigned state = 0, set = 1;   // bump/transparency maps always end up on texcoord1 in the original
    if (!m || !varUints(findVar(m, "componentState"), &state, 1) || !(state & bit)) return nullptr;
    TexSampler* t = nullptr;
    Buffer* b = textureBuffer(findVar(m, var), &t);
    if (!b) return nullptr;
    if (setVar) varUints(findVar(m, setVar), &set, 1);
    *sampler = (t->wrap[0] & 3) | ((t->wrap[1] & 3) << 2) | (t->filter[1] == 1 ? DGB_SAMPLER_LINEAR : 0) |
               ((set & 1) << 8);
    return b;
}

static Buffer* diffuseTextureOf(const Material* m, uint32_t* sampler)
{
    return phongTexture(m, "diffuseTexture", 4, "diffuseTexCoordID", sampler);
}

static void fillMaterial(const Material* m, DgbInstance& di)
{
    di.base[0] = di.base[1] = di.base[2] = 0.7f;
    di.reflMin = di.reflMax = 0.0f;
    di.kind = DGB_MAT_OTHER;
    di.shininess = 1.0f;
    di.ior = 1.5f;
    std::memcpy(di.base2, di.base, sizeof(di.base2));
    if (!m) return;
    float c[4] = { 0, 0, 0, 0 };
    auto setBase = [&](const char* name) {
        if (varFloats(findVar(m, name), c, 3) == 3) std::memcpy(di.base, c, 12);
    };
    auto setColor = [&](float* dst, const char* name) {
        if (varFloats(findVar(m, name), c, 3) == 3) std::memcpy(dst, c, 12);
    };
    auto getFloat = [&](const char* name, float def) { return varFloats(findVar(m, name), c, 1) ? c[0] : def; };
    auto setRefl = [&](float defMin, float defMax) {
        di.reflMin = varFloats(findVar(m, "reflectivityMin"), c, 1) ? c[0] : defMin;
        di.reflMax = varFloats(findVar(m, "reflectivityMax"), c, 1) ? c[0] : defMax;
    };
    // Defaults below are the defaults declared in the demo's own shaders.
    if (findVar(m, "diffuseColor0")) {
        di.kind = DGB_MAT_CARPAINT;
        di.base[0] = 1.0f; di.base[1] = di.base[2] = 0.0f;
        setBase("diffuseColor0");
        di.base2[0] = 1.0f; di.base2[1] = di.base2[2] = 0.0f;
        setColor(di.base2, "diffuseColor1");
        di.specular[0] = di.specular[1] = di.specular[2] = 0.2f;
        setColor(di.specular, "specularColor");
        setRefl(0.2f, 0.7f);
        di.reflectivityFalloff = getFloat("reflectivityFalloff", 1.0f);
        di.glossyColor[0] = di.glossyColor[1] = 0.1f; di.glossyColor[2] = 0.0f;
        setColor(di.glossyColor, "glossyColor");
        di.glossiness = getFloat("glossiness", 0.1f);
    } else if (findVar(m, "refractionIndex")) {
        di.kind = DGB_MAT_GLASS;
        di.base[0] = di.base[1] = di.base[2] = 0.99f;
        setBase("refractionColor");
        di.reflectionColor[0] = di.reflectionColor[1] = di.reflectionColor[2] = 0.99f;
        setColor(di.reflectionColor, "reflectionColor");
        di.ior = getFloat("refractionIndex", 1.93f);
    } else if (findVar(m, "shininess_u")) {
        di.kind = DGB_MAT_METAL;
        di.base[0] = di.base[1] = di.base[2] = 1.0f;
        setBase("specularColor");
        std::memcpy(di.specular, di.base, sizeof(di.specular));
        di.shininessU = getFloat("shininess_u", 10.0f);
        di.shininessV = getFloat("shininess_v", 100.0f);
        di.absorption = getFloat("absorption", 3.433f);
        di.shininess = 0.5f * (di.shininessU + di.shininessV);
    } else if (findVar(m, "diffuseColor")) {
        di.kind = DGB_MAT_PHONG;
        setBase("diffuseColor");
        setColor(di.specular, "specularColor");
        di.shininess = getFloat("shininess", 1.0f);
        setRefl(0.0f, 0.0f);
        uint32_t sampler = 0;
        if (Buffer* tb = diffuseTextureOf(m, &sampler)) {
            di.diffuseTex = textureKey(tb);
            di.diffuseSampler = sampler;
        }
        varUints(findVar(m, "componentState"), &di.componentState, 1);
        di.transparency = getFloat("transparency", 0.0f);
        di.transparentColor[0] = di.transparentColor[1] = di.transparentColor[2] = 0.0f;
        di.transparentColor[3] = 1.0f;
        varFloats(findVar(m, "transparentColor"), di.transparentColor, 4);
        di.bumpScale = getFloat("bumpScale", 1.0f);
        if (Buffer* tb = phongTexture(m, "bumpTexture", 1024, nullptr, &sampler)) {
            di.bumpTex = textureKey(tb);
            di.bumpSampler = sampler;
        }
        if (Buffer* tb = phongTexture(m, "transparentTexture", 16, nullptr, &sampler)) {
            di.transparentTex = textureKey(tb);
            di.transparentSampler = sampler;
        }
    }
}

static bool bridgeSync(Context* c)
{
    Obj* top = findTopObject(c);
    static const float I[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    std::vector<SceneInst> insts;
    collect(top, I, insts, 0);

    std::vector<MeshSrc> srcs(insts.size());
    std::vector<bool> valid(insts.size());
    unsigned newMeshes = 0, skipped = 0, withNormals = 0;
    g_br.fill = 0;
    std::set<unsigned long long> pending;
    for (size_t i = 0; i < insts.size(); ++i) {
        valid[i] = meshSource(insts[i].geo, srcs[i]);
        if (!valid[i]) { ++skipped; continue; }
        if (srcs[i].flags & DGB_MESH_NORMALS) ++withNormals;
        if (g_br.sentMeshes.count(srcs[i].key) || pending.count(srcs[i].key)) continue;
        if (!writeMesh(srcs[i])) { g_br.sentMeshes.clear(); return false; }
        pending.insert(srcs[i].key);
        ++newMeshes;
    }

    unsigned newTextures = 0;
    std::set<unsigned long long> pendingTex;
    for (size_t i = 0; i < insts.size(); ++i) {
        if (!valid[i]) continue;
        uint32_t sampler = 0;
        Buffer* used[3] = { diffuseTextureOf(insts[i].mat, &sampler),
                            phongTexture(insts[i].mat, "bumpTexture", 1024, nullptr, &sampler),
                            phongTexture(insts[i].mat, "transparentTexture", 16, nullptr, &sampler) };
        for (Buffer* tb : used) {
            if (!tb) continue;
            const unsigned long long key = textureKey(tb);
            if (g_br.sentTextures.count(key) || pendingTex.count(key)) continue;
            if (!writeTexture(tb, key)) { g_br.sentMeshes.clear(); g_br.sentTextures.clear(); return false; }
            pendingTex.insert(key);
            ++newTextures;
        }
    }

    std::vector<DgbInstance> out;
    for (size_t i = 0; i < insts.size(); ++i) {
        if (!valid[i]) continue;
        DgbInstance di = {};
        di.meshKey = srcs[i].key;
        std::memcpy(di.xform, insts[i].M, sizeof(di.xform));   // first 3 rows of the row-major 4x4
        fillMaterial(insts[i].mat, di);
        out.push_back(di);
    }
    const DgbRecord rec = { DGB_REC_INSTANCES, 0 };
    const uint32_t countPad[2] = { (uint32_t)out.size(), 0 };
    const DgbRecord end = { DGB_REC_END, 0 };
    if (!putBytes(&rec, sizeof(rec)) || !putBytes(countPad, sizeof(countPad)) ||
        (!out.empty() && !putBytes(out.data(), out.size() * sizeof(DgbInstance))) || !putBytes(&end, sizeof(end)) ||
        !flushChunk() || !bridgeCall(DGB_SYNC, 300000)) {
        g_br.sentMeshes.clear();   // the helper discards a failed stream; resend everything next time
        g_br.sentTextures.clear();
        return false;
    }
    g_br.sentMeshes.insert(pending.begin(), pending.end());
    g_br.sentTextures.insert(pendingTex.begin(), pendingTex.end());
    unsigned textured = 0;
    for (const DgbInstance& di : out) textured += di.diffuseTex != 0;
    logLine("bridge: scene sent from %s: %u instances (%u with smooth normals, %u textured, %u skipped), "
            "%u new meshes, %u new textures",
            N(top), (unsigned)out.size(), withNormals, textured, skipped, newMeshes, newTextures);
    return true;
}

static Buffer* findOutputBuffer(Context* c)
{
    Var* v = findVar(c, "outputBuffer");
    if (v && v->object && v->object->kind == K_BUFFER && ((Buffer*)v->object)->glId) return (Buffer*)v->object;
    for (Obj* o : g_all)
        if (o->kind == K_BUFFER && !o->destroyed && o->ctx == c && ((Buffer*)o)->glId &&
            (((Buffer*)o)->desc & BUF_OUTPUT))
            return (Buffer*)o;
    return nullptr;
}

static bool bridgeRender(Context* c)
{
    static bool loggedFormat = false;
    Buffer* out = findOutputBuffer(c);
    if (!out) return false;
    if (out->format != FMT_FLOAT + 3) {
        if (!loggedFormat) logLine("bridge: output buffer format %s not supported yet", formatName(out->format).c_str());
        loggedFormat = true;
        return false;
    }
    DgbRenderParams rp = {};
    rp.width = (unsigned)out->dims[0];
    rp.height = (unsigned)out->dims[1];
    float f[4];
    int fn = 0;
    if (Var* v = findVar(c, "frameNumber")) if (v->data.size() >= 4) std::memcpy(&fn, v->data.data(), 4);
    rp.frameNumber = (uint32_t)fn;
    varFloats(findVar(c, "eye"), rp.eye, 3);
    varFloats(findVar(c, "U"), rp.U, 3);
    varFloats(findVar(c, "V"), rp.V, 3);
    varFloats(findVar(c, "W"), rp.W, 3);
    rp.sunDir[1] = 1.0f;
    if (varFloats(findVar(c, "sunDirection"), f, 3) == 3) {
        const float len = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
        if (len > 0) for (int k = 0; k < 3; ++k) rp.sunDir[k] = f[k] / len;
    }
    varFloats(findVar(c, "sunColor"), rp.sunColor, 3);          // physical units, as the demo sets it
    if (varFloats(findVar(c, "overcast"), f, 1)) rp.overcast = std::min(std::max(f[0], 0.0f), 1.0f);
    if (varFloats(findVar(c, "camLensRadius"), f, 1)) rp.lensRadius = std::max(f[0], 0.0f);
    varFloats(findVar(c, "camFocalDistance"), &rp.focalDistance, 1);
    static const char* skyNames[5] = { "c0", "c1", "c2", "c3", "c4" };
    for (int k = 0; k < 5; ++k) varFloats(findVar(c, skyNames[k]), rp.skyC[k], 3);
    varFloats(findVar(c, "inv_divisor_Yxy"), rp.skyInvDivisorYxy, 3);
    int depth = 5;
    if (Var* v = findVar(c, "maxReflectionDepth")) if (v->data.size() >= 4) std::memcpy(&depth, v->data.data(), 4);
    rp.maxDepth = (uint32_t)std::max(depth, 1);
    const float wlen = std::sqrt(rp.W[0] * rp.W[0] + rp.W[1] * rp.W[1] + rp.W[2] * rp.W[2]);
    rp.epsilon = (varFloats(findVar(c, "sceneEpsilon"), f, 1) && f[0] > 0) ? f[0] : std::max(wlen * 1e-4f, 1e-4f);

    std::memcpy(g_br.data, &rp, sizeof(rp));
    if (!bridgeCall(DGB_RENDER, 20000)) return false;
    const size_t bytes = (size_t)g_br.hdr->arg[0];
    // DGBRIDGE_TEST_NOGL=1 (testing only): read the image back without OpenGL.
    static const bool testNoGL = envFlag("DGBRIDGE_TEST_NOGL", false);
    const bool gl = glReady();
    if (!gl && !testNoGL) return false;

    int prev = 0;
    const int glSize = gl ? glBindForUpload(out->glId, &prev) : (int)bytes;
    const size_t total = std::min(bytes, (size_t)glSize);
    bool ok = true;
    for (size_t off = 0; off < total && ok; off += DGB_SHM_DATA_SIZE) {
        const size_t len = std::min((size_t)DGB_SHM_DATA_SIZE, total - off);
        g_br.hdr->arg[0] = off;
        g_br.hdr->arg[1] = len;
        ok = bridgeCall(DGB_READ, 20000);
        if (ok && gl) g_gl.bufferSubData(GL_ARRAY_BUFFER_, (ptrdiff_t)off, (ptrdiff_t)len, g_br.data);
        if (ok && !gl && off == 0) {
            const float* px = (const float*)g_br.data;
            logLine("bridge (test): first pixel %g %g %g %g", px[0], px[1], px[2], px[3]);
        }
    }
    if (gl) g_gl.bindBuffer(GL_ARRAY_BUFFER_, (unsigned)prev);
    if (ok && (++g_br.frames <= 3 || g_br.frames % 500 == 0))
        logLine("bridge: frame %llu delivered (%ux%u, %u bytes into GL buffer %u)", g_br.frames, rp.width, rp.height,
                (unsigned)total, out->glId);
    return ok;
}

// Called on every launch: keeps the helper's scene up to date and renders.
static bool bridgeLaunch(Context* c)
{
    if (!g_br.tried) bridgeStart();
    if (!g_br.ok) return false;
    if (g_br.syncedStruct != g_structVersion || g_br.syncedMaterial != g_materialVersion) {
        // Mark as synced even on failure so a broken scene is not resent every frame.
        g_br.syncedStruct = g_structVersion;
        g_br.syncedMaterial = g_materialVersion;
        if (!bridgeSync(c)) return false;
    }
    return bridgeRender(c);
}

// ---------------------------------------------------------------------------
// 4x4 matrix inverse (row-major)
// ---------------------------------------------------------------------------

static bool invert4x4(const float* m, float* out)
{
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (std::fabs(det) < 1e-30f) return false;
    det = 1.0f / det;
    for (int i = 0; i < 16; ++i) out[i] = inv[i] * det;
    return true;
}

static void transpose4x4(const float* in, float* out)
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) out[c * 4 + r] = in[r * 4 + c];
}

// ===========================================================================
// API: version and devices
// ===========================================================================

API RTresult rtGetVersion(unsigned* version)
{
    ENTER("rtGetVersion");
    if (!version) return R(_cs, RT_ERROR_INVALID_VALUE, "rtGetVersion(NULL)");
    *version = 3000;   // value returned by the real OptiX 3.0.0 DLL
    return R(_cs, RT_SUCCESS, "rtGetVersion() = 3000");
}

API RTresult rtDeviceGetDeviceCount(unsigned* count)
{
    ENTER("rtDeviceGetDeviceCount");
    if (!count) return R(_cs, RT_ERROR_INVALID_VALUE, "rtDeviceGetDeviceCount(NULL)");
    *count = 1;
    return R(_cs, RT_SUCCESS, "rtDeviceGetDeviceCount() = 1");
}

API RTresult rtDeviceGetAttribute(int ordinal, unsigned attrib, RTsize size, void* p)
{
    ENTER("rtDeviceGetAttribute");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtDeviceGetAttribute(%d, %u, %u, NULL)", ordinal, attrib, (unsigned)size);
    if (ordinal != 0) return R(_cs, RT_ERROR_INVALID_DEVICE, "rtDeviceGetAttribute(%d, %u)", ordinal, attrib);
    std::memset(p, 0, size);
    int iv = 0;
    const char* what = "";
    switch (attrib) {
    case 0: iv = 1024; what = "MAX_THREADS_PER_BLOCK"; break;
    case 1: iv = 2500000; what = "CLOCK_RATE"; break;
    case 2: iv = 70; what = "MULTIPROCESSOR_COUNT"; break;
    case 3: iv = 0; what = "EXECUTION_TIMEOUT_ENABLED"; break;
    case 4: iv = 128; what = "MAX_HARDWARE_TEXTURE_COUNT"; break;
    case 5: {
        const char* name = "DG Bridge virtual device";
        std::strncpy((char*)p, name, size ? size - 1 : 0);
        return R(_cs, RT_SUCCESS, "rtDeviceGetAttribute(0, NAME, %u) = \"%s\"", (unsigned)size, name);
    }
    case 6: {
        int cc[2] = { 3, 0 };   // report a Kepler-class device
        std::memcpy(p, cc, std::min(size, sizeof(cc)));
        return R(_cs, RT_SUCCESS, "rtDeviceGetAttribute(0, COMPUTE_CAPABILITY, %u) = 3.0", (unsigned)size);
    }
    case 7: {
        // The demo reads this as a 4-byte value; stay below 2 GB so that
        // neither a signed nor an unsigned comparison can misbehave.
        RTsize mem = 0x7FF00000u;
        std::memcpy(p, &mem, std::min(size, sizeof(mem)));
        return R(_cs, RT_SUCCESS, "rtDeviceGetAttribute(0, TOTAL_MEMORY, %u) = 0x%x", (unsigned)size, (unsigned)mem);
    }
    case 8: iv = 0; what = "TCC_DRIVER"; break;
    case 9: iv = 0; what = "CUDA_DEVICE_ORDINAL"; break;
    default:
        return R(_cs, RT_ERROR_INVALID_VALUE, "rtDeviceGetAttribute(0, unknown attribute %u, %u)", attrib, (unsigned)size);
    }
    std::memcpy(p, &iv, std::min(size, sizeof(iv)));
    return R(_cs, RT_SUCCESS, "rtDeviceGetAttribute(0, %s, %u) = %d", what, (unsigned)size, iv);
}

// ===========================================================================
// API: generic object functions
// ===========================================================================

template <class T>
static RTresult createObj(const char* fn, Kind k, void* hc, void** out)
{
    ENTER(fn);
    Context* c = as<Context>(hc, K_CONTEXT);
    if (!c) return badHandle(_cs, fn, K_CONTEXT, hc);
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "%s(%s, NULL)", fn, N(c));
    T* o = mk<T>(k, c);
    *out = o;
    return R(_cs, RT_SUCCESS, "%s(%s) = %s", fn, N(c), N(o));
}

static RTresult destroyObj(const char* fn, Kind k, void* h)
{
    ENTER(fn);
    Obj* o = as<Obj>(h, k);
    if (!o) return badHandle(_cs, fn, k, h);
    if (k == K_CONTEXT) dumpScene((Context*)o, "context destroyed");
    o->destroyed = true;
    return R(_cs, RT_SUCCESS, "%s(%s)", fn, N(o));
}

static RTresult validateObj(const char* fn, Kind k, void* h)
{
    ENTER(fn);
    Obj* o = as<Obj>(h, k);
    if (!o) return badHandle(_cs, fn, k, h);
    return R(_cs, RT_SUCCESS, "%s(%s)", fn, N(o));
}

static RTresult getContextObj(const char* fn, Kind k, void* h, void** out)
{
    ENTER(fn);
    Obj* o = as<Obj>(h, k);
    if (!o) return badHandle(_cs, fn, k, h);
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "%s(%s, NULL)", fn, N(o));
    *out = o->ctx;
    return R(_cs, RT_SUCCESS, "%s(%s) = %s", fn, N(o), N(o->ctx));
}

#define COMMON_NO_CREATE(Prefix, KIND)                                                            \
    API RTresult rt##Prefix##Destroy(void* h) { return destroyObj("rt" #Prefix "Destroy", KIND, h); } \
    API RTresult rt##Prefix##Validate(void* h) { return validateObj("rt" #Prefix "Validate", KIND, h); } \
    API RTresult rt##Prefix##GetContext(void* h, void** c) { return getContextObj("rt" #Prefix "GetContext", KIND, h, c); }

#define COMMON_FUNCS(Prefix, T, KIND)                                                             \
    API RTresult rt##Prefix##Create(void* c, void** out) { return createObj<T>("rt" #Prefix "Create", KIND, c, out); } \
    COMMON_NO_CREATE(Prefix, KIND)

COMMON_FUNCS(Group, Group, K_GROUP)
COMMON_FUNCS(Selector, Selector, K_SELECTOR)
COMMON_FUNCS(Transform, Transform, K_TRANSFORM)
COMMON_FUNCS(GeometryGroup, GeomGroup, K_GEOMGROUP)
COMMON_FUNCS(Acceleration, Accel, K_ACCEL)
COMMON_FUNCS(GeometryInstance, GInst, K_GINST)
COMMON_FUNCS(Geometry, Geometry, K_GEOMETRY)
COMMON_FUNCS(Material, Material, K_MATERIAL)
COMMON_FUNCS(TextureSampler, TexSampler, K_TEXSAMPLER)
COMMON_NO_CREATE(Program, K_PROGRAM)
COMMON_NO_CREATE(Buffer, K_BUFFER)

API RTresult rtVariableGetContext(void* h, void** c) { return getContextObj("rtVariableGetContext", K_VARIABLE, h, c); }

// ===========================================================================
// API: variable scopes
// ===========================================================================

static RTresult scopeDeclare(const char* fn, Kind k, void* h, const char* name, void** out)
{
    ENTER(fn);
    Scope* s = as<Scope>(h, k);
    if (!s) return badHandle(_cs, fn, k, h);
    if (!name || !out) return R(_cs, RT_ERROR_INVALID_VALUE, "%s(%s, %s, %p)", fn, N(s), name ? name : "NULL", (void*)out);
    for (Var* v : s->vars)
        if (!v->destroyed && v->name == name)
            return R(_cs, RT_ERROR_VARIABLE_REDECLARED, "%s(%s, \"%s\")", fn, N(s), name);
    Var* v = mk<Var>(K_VARIABLE, s->ctx);
    v->name = name;
    v->owner = s;
    s->vars.push_back(v);
    *out = v;
    return R(_cs, RT_SUCCESS, "%s(%s, \"%s\") = %s", fn, N(s), name, N(v));
}

static RTresult scopeQuery(const char* fn, Kind k, void* h, const char* name, void** out)
{
    ENTER(fn);
    Scope* s = as<Scope>(h, k);
    if (!s) return badHandle(_cs, fn, k, h);
    if (!name || !out) return R(_cs, RT_ERROR_INVALID_VALUE, "%s(%s, %s, %p)", fn, N(s), name ? name : "NULL", (void*)out);
    *out = nullptr;
    for (Var* v : s->vars)
        if (!v->destroyed && v->name == name) {
            *out = v;
            break;
        }
    return R(_cs, RT_SUCCESS, "%s(%s, \"%s\") = %s", fn, N(s), name, N(*out));
}

static RTresult scopeRemove(const char* fn, Kind k, void* h, void* hv)
{
    ENTER(fn);
    Scope* s = as<Scope>(h, k);
    if (!s) return badHandle(_cs, fn, k, h);
    Var* v = as<Var>(hv, K_VARIABLE);
    auto it = std::find(s->vars.begin(), s->vars.end(), v);
    if (!v || it == s->vars.end()) return R(_cs, RT_ERROR_VARIABLE_NOT_FOUND, "%s(%s, %s)", fn, N(s), N(hv));
    s->vars.erase(it);
    v->destroyed = true;
    return R(_cs, RT_SUCCESS, "%s(%s, %s \"%s\")", fn, N(s), N(v), v->name.c_str());
}

static RTresult scopeCount(const char* fn, Kind k, void* h, unsigned* out)
{
    ENTER(fn);
    Scope* s = as<Scope>(h, k);
    if (!s) return badHandle(_cs, fn, k, h);
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "%s(%s, NULL)", fn, N(s));
    *out = (unsigned)s->vars.size();
    return R(_cs, RT_SUCCESS, "%s(%s) = %u", fn, N(s), *out);
}

static RTresult scopeGet(const char* fn, Kind k, void* h, unsigned index, void** out)
{
    ENTER(fn);
    Scope* s = as<Scope>(h, k);
    if (!s) return badHandle(_cs, fn, k, h);
    if (!out || index >= s->vars.size()) return R(_cs, RT_ERROR_INVALID_VALUE, "%s(%s, %u)", fn, N(s), index);
    *out = s->vars[index];
    return R(_cs, RT_SUCCESS, "%s(%s, %u) = %s \"%s\"", fn, N(s), index, N(*out), s->vars[index]->name.c_str());
}

#define SCOPE_FUNCS(Prefix, KIND)                                                                                       \
    API RTresult rt##Prefix##DeclareVariable(void* h, const char* n, void** v) { return scopeDeclare("rt" #Prefix "DeclareVariable", KIND, h, n, v); } \
    API RTresult rt##Prefix##QueryVariable(void* h, const char* n, void** v) { return scopeQuery("rt" #Prefix "QueryVariable", KIND, h, n, v); }       \
    API RTresult rt##Prefix##RemoveVariable(void* h, void* v) { return scopeRemove("rt" #Prefix "RemoveVariable", KIND, h, v); }                       \
    API RTresult rt##Prefix##GetVariableCount(void* h, unsigned* c) { return scopeCount("rt" #Prefix "GetVariableCount", KIND, h, c); }               \
    API RTresult rt##Prefix##GetVariable(void* h, unsigned i, void** v) { return scopeGet("rt" #Prefix "GetVariable", KIND, h, i, v); }

SCOPE_FUNCS(Context, K_CONTEXT)
SCOPE_FUNCS(Program, K_PROGRAM)
SCOPE_FUNCS(Selector, K_SELECTOR)
SCOPE_FUNCS(GeometryInstance, K_GINST)
SCOPE_FUNCS(Geometry, K_GEOMETRY)
SCOPE_FUNCS(Material, K_MATERIAL)

// ===========================================================================
// API: children (Group, Selector, GeometryGroup)
// ===========================================================================

template <class T>
static RTresult childSetCount(const char* fn, Kind k, void* h, unsigned count)
{
    ENTER(fn);
    T* o = as<T>(h, k);
    if (!o) return badHandle(_cs, fn, k, h);
    o->children.resize(count, nullptr);
    return R(_cs, RT_SUCCESS, "%s(%s, %u)", fn, N(o), count);
}

template <class T>
static RTresult childGetCount(const char* fn, Kind k, void* h, unsigned* out)
{
    ENTER(fn);
    T* o = as<T>(h, k);
    if (!o) return badHandle(_cs, fn, k, h);
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "%s(%s, NULL)", fn, N(o));
    *out = (unsigned)o->children.size();
    return R(_cs, RT_SUCCESS, "%s(%s) = %u", fn, N(o), *out);
}

template <class T>
static RTresult childSet(const char* fn, Kind k, void* h, unsigned index, void* child)
{
    ENTER(fn);
    T* o = as<T>(h, k);
    if (!o) return badHandle(_cs, fn, k, h);
    Obj* c = asAny(child);
    if (!c || index >= o->children.size())
        return R(_cs, RT_ERROR_INVALID_VALUE, "%s(%s, %u, %s) [count %u]", fn, N(o), index, N(child), (unsigned)o->children.size());
    o->children[index] = c;
    return R(_cs, RT_SUCCESS, "%s(%s, %u, %s)", fn, N(o), index, N(c));
}

template <class T>
static RTresult childGet(const char* fn, Kind k, void* h, unsigned index, void** out)
{
    ENTER(fn);
    T* o = as<T>(h, k);
    if (!o) return badHandle(_cs, fn, k, h);
    if (!out || index >= o->children.size()) return R(_cs, RT_ERROR_INVALID_VALUE, "%s(%s, %u)", fn, N(o), index);
    *out = o->children[index];
    return R(_cs, RT_SUCCESS, "%s(%s, %u) = %s", fn, N(o), index, N(*out));
}

template <class T>
static RTresult childGetType(const char* fn, Kind k, void* h, unsigned index, unsigned* out)
{
    ENTER(fn);
    T* o = as<T>(h, k);
    if (!o) return badHandle(_cs, fn, k, h);
    if (!out || index >= o->children.size()) return R(_cs, RT_ERROR_INVALID_VALUE, "%s(%s, %u)", fn, N(o), index);
    *out = objectTypeOf(o->children[index]);
    return R(_cs, RT_SUCCESS, "%s(%s, %u) = %s", fn, N(o), index, typeName(*out));
}

#define CHILD_FUNCS(Prefix, T, KIND)                                                                                   \
    API RTresult rt##Prefix##SetChildCount(void* h, unsigned n) { return childSetCount<T>("rt" #Prefix "SetChildCount", KIND, h, n); }  \
    API RTresult rt##Prefix##GetChildCount(void* h, unsigned* n) { return childGetCount<T>("rt" #Prefix "GetChildCount", KIND, h, n); } \
    API RTresult rt##Prefix##SetChild(void* h, unsigned i, void* c) { return childSet<T>("rt" #Prefix "SetChild", KIND, h, i, c); }    \
    API RTresult rt##Prefix##GetChild(void* h, unsigned i, void** c) { return childGet<T>("rt" #Prefix "GetChild", KIND, h, i, c); }

CHILD_FUNCS(Group, Group, K_GROUP)
CHILD_FUNCS(Selector, Selector, K_SELECTOR)
CHILD_FUNCS(GeometryGroup, GeomGroup, K_GEOMGROUP)

API RTresult rtGroupGetChildType(void* h, unsigned i, unsigned* t) { return childGetType<Group>("rtGroupGetChildType", K_GROUP, h, i, t); }
API RTresult rtSelectorGetChildType(void* h, unsigned i, unsigned* t) { return childGetType<Selector>("rtSelectorGetChildType", K_SELECTOR, h, i, t); }

// Accelerations attached to groups
template <class T>
static RTresult setAccelOf(const char* fn, Kind k, void* h, void* ha)
{
    ENTER(fn);
    T* o = as<T>(h, k);
    if (!o) return badHandle(_cs, fn, k, h);
    Accel* a = as<Accel>(ha, K_ACCEL);
    if (!a) return badHandle(_cs, fn, K_ACCEL, ha);
    o->accel = a;
    return R(_cs, RT_SUCCESS, "%s(%s, %s)", fn, N(o), N(a));
}

template <class T>
static RTresult getAccelOf(const char* fn, Kind k, void* h, void** out)
{
    ENTER(fn);
    T* o = as<T>(h, k);
    if (!o) return badHandle(_cs, fn, k, h);
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "%s(%s, NULL)", fn, N(o));
    *out = o->accel;
    return R(_cs, RT_SUCCESS, "%s(%s) = %s", fn, N(o), N(o->accel));
}

API RTresult rtGroupSetAcceleration(void* h, void* a) { return setAccelOf<Group>("rtGroupSetAcceleration", K_GROUP, h, a); }
API RTresult rtGroupGetAcceleration(void* h, void** a) { return getAccelOf<Group>("rtGroupGetAcceleration", K_GROUP, h, a); }
API RTresult rtGeometryGroupSetAcceleration(void* h, void* a) { return setAccelOf<GeomGroup>("rtGeometryGroupSetAcceleration", K_GEOMGROUP, h, a); }
API RTresult rtGeometryGroupGetAcceleration(void* h, void** a) { return getAccelOf<GeomGroup>("rtGeometryGroupGetAcceleration", K_GEOMGROUP, h, a); }

// ===========================================================================
// API: Context
// ===========================================================================

API RTresult rtContextCreate(void** out)
{
    ENTER("rtContextCreate");
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "rtContextCreate(NULL)");
    Context* c = mk<Context>(K_CONTEXT, nullptr);
    c->ctx = c;
    *out = c;
    return R(_cs, RT_SUCCESS, "rtContextCreate() = %s", N(c));
}

API RTresult rtContextDestroy(void* h) { return destroyObj("rtContextDestroy", K_CONTEXT, h); }
API RTresult rtContextValidate(void* h) { return validateObj("rtContextValidate", K_CONTEXT, h); }

API void rtContextGetErrorString(void* h, RTresult code, const char** out)
{
    ENTER("rtContextGetErrorString");
    static std::string s;
    s = resultName(code);
    if (code != RT_SUCCESS && !g_lastError.empty()) s += " (DG Bridge: " + g_lastError + ")";
    if (out) *out = s.c_str();
    R(_cs, RT_SUCCESS, "rtContextGetErrorString(%s, 0x%x) = \"%s\"", N(h), code, s.c_str());
}

#define GET_CTX(fn)                                    \
    ENTER(fn);                                         \
    Context* c = as<Context>(h, K_CONTEXT);            \
    if (!c) return badHandle(_cs, fn, K_CONTEXT, h)

API RTresult rtContextSetAttribute(void* h, unsigned attrib, RTsize size, const void* p)
{
    GET_CTX("rtContextSetAttribute");
    return R(_cs, RT_SUCCESS, "rtContextSetAttribute(%s, 0x%x, %u, %s) [ignored]", N(c), attrib,
             (unsigned)size, p ? fmtHex(p, size).c_str() : "NULL");
}

API RTresult rtContextGetAttribute(void* h, unsigned attrib, RTsize size, void* p)
{
    GET_CTX("rtContextGetAttribute");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtContextGetAttribute(%s, 0x%x, NULL)", N(c), attrib);
    std::memset(p, 0, size);
    if (attrib == 0) {             // MAX_TEXTURE_COUNT
        int v = 128;
        std::memcpy(p, &v, std::min(size, sizeof(v)));
    } else if (attrib == 1) {      // CPU_NUM_THREADS
        int v = 4;
        std::memcpy(p, &v, std::min(size, sizeof(v)));
    } else if (attrib >= 0x10000000) {   // AVAILABLE_DEVICE_MEMORY + ordinal
        RTsize v = 0x7FF00000u;
        std::memcpy(p, &v, std::min(size, sizeof(v)));
    }
    return R(_cs, RT_SUCCESS, "rtContextGetAttribute(%s, 0x%x, %u) = %s", N(c), attrib, (unsigned)size,
             fmtHex(p, size).c_str());
}

API RTresult rtContextSetDevices(void* h, unsigned count, const int* devices)
{
    GET_CTX("rtContextSetDevices");
    if (count && !devices) return R(_cs, RT_ERROR_INVALID_VALUE, "rtContextSetDevices(%s, %u, NULL)", N(c), count);
    c->devices.assign(devices, devices + count);
    return R(_cs, RT_SUCCESS, "rtContextSetDevices(%s, %u, %s)", N(c), count, fmtInts(devices, (int)count).c_str());
}

API RTresult rtContextGetDevices(void* h, int* devices)
{
    GET_CTX("rtContextGetDevices");
    if (!devices) return R(_cs, RT_ERROR_INVALID_VALUE, "rtContextGetDevices(%s, NULL)", N(c));
    std::copy(c->devices.begin(), c->devices.end(), devices);
    return R(_cs, RT_SUCCESS, "rtContextGetDevices(%s) = %s", N(c), fmtInts(c->devices.data(), (int)c->devices.size()).c_str());
}

API RTresult rtContextGetDeviceCount(void* h, unsigned* count)
{
    GET_CTX("rtContextGetDeviceCount");
    if (!count) return R(_cs, RT_ERROR_INVALID_VALUE, "rtContextGetDeviceCount(%s, NULL)", N(c));
    *count = (unsigned)c->devices.size();
    return R(_cs, RT_SUCCESS, "rtContextGetDeviceCount(%s) = %u", N(c), *count);
}

API RTresult rtContextSetStackSize(void* h, RTsize bytes)
{
    GET_CTX("rtContextSetStackSize");
    c->stackSize = bytes;
    return R(_cs, RT_SUCCESS, "rtContextSetStackSize(%s, %u)", N(c), (unsigned)bytes);
}

API RTresult rtContextGetStackSize(void* h, RTsize* bytes)
{
    GET_CTX("rtContextGetStackSize");
    if (!bytes) return R(_cs, RT_ERROR_INVALID_VALUE, "rtContextGetStackSize(%s, NULL)", N(c));
    *bytes = c->stackSize;
    return R(_cs, RT_SUCCESS, "rtContextGetStackSize(%s) = %u", N(c), (unsigned)*bytes);
}

API RTresult rtContextSetTimeoutCallback(void* h, void* callback, double minPollingSeconds)
{
    GET_CTX("rtContextSetTimeoutCallback");
    c->timeoutCallback = callback;
    c->timeoutSeconds = minPollingSeconds;
    return R(_cs, RT_SUCCESS, "rtContextSetTimeoutCallback(%s, %p, %g)", N(c), callback, minPollingSeconds);
}

API RTresult rtContextSetEntryPointCount(void* h, unsigned n)
{
    GET_CTX("rtContextSetEntryPointCount");
    c->entryCount = n;
    return R(_cs, RT_SUCCESS, "rtContextSetEntryPointCount(%s, %u)", N(c), n);
}

API RTresult rtContextGetEntryPointCount(void* h, unsigned* n)
{
    GET_CTX("rtContextGetEntryPointCount");
    if (!n) return R(_cs, RT_ERROR_INVALID_VALUE, "rtContextGetEntryPointCount(%s, NULL)", N(c));
    *n = c->entryCount;
    return R(_cs, RT_SUCCESS, "rtContextGetEntryPointCount(%s) = %u", N(c), *n);
}

API RTresult rtContextSetRayTypeCount(void* h, unsigned n)
{
    GET_CTX("rtContextSetRayTypeCount");
    c->rayTypeCount = n;
    return R(_cs, RT_SUCCESS, "rtContextSetRayTypeCount(%s, %u)", N(c), n);
}

API RTresult rtContextGetRayTypeCount(void* h, unsigned* n)
{
    GET_CTX("rtContextGetRayTypeCount");
    if (!n) return R(_cs, RT_ERROR_INVALID_VALUE, "rtContextGetRayTypeCount(%s, NULL)", N(c));
    *n = c->rayTypeCount;
    return R(_cs, RT_SUCCESS, "rtContextGetRayTypeCount(%s) = %u", N(c), *n);
}

static RTresult setProgSlot(const CallScope& cs, const char* fn, Context* c, std::map<unsigned, Obj*>& slots,
                            unsigned index, void* hp)
{
    Program* p = as<Program>(hp, K_PROGRAM);
    if (!p && hp) return badHandle(cs, fn, K_PROGRAM, hp);
    slots[index] = p;
    return R(cs, RT_SUCCESS, "%s(%s, %u, %s \"%s\")", fn, N(c), index, N(p), p ? p->name.c_str() : "");
}

static RTresult getProgSlot(const CallScope& cs, const char* fn, Context* c, std::map<unsigned, Obj*>& slots,
                            unsigned index, void** out)
{
    if (!out) return R(cs, RT_ERROR_INVALID_VALUE, "%s(%s, %u, NULL)", fn, N(c), index);
    auto it = slots.find(index);
    *out = it == slots.end() ? nullptr : it->second;
    return R(cs, RT_SUCCESS, "%s(%s, %u) = %s", fn, N(c), index, N(*out));
}

API RTresult rtContextSetRayGenerationProgram(void* h, unsigned i, void* p)
{
    GET_CTX("rtContextSetRayGenerationProgram");
    return setProgSlot(_cs, "rtContextSetRayGenerationProgram", c, c->raygen, i, p);
}
API RTresult rtContextGetRayGenerationProgram(void* h, unsigned i, void** p)
{
    GET_CTX("rtContextGetRayGenerationProgram");
    return getProgSlot(_cs, "rtContextGetRayGenerationProgram", c, c->raygen, i, p);
}
API RTresult rtContextSetExceptionProgram(void* h, unsigned i, void* p)
{
    GET_CTX("rtContextSetExceptionProgram");
    return setProgSlot(_cs, "rtContextSetExceptionProgram", c, c->exceptionProg, i, p);
}
API RTresult rtContextGetExceptionProgram(void* h, unsigned i, void** p)
{
    GET_CTX("rtContextGetExceptionProgram");
    return getProgSlot(_cs, "rtContextGetExceptionProgram", c, c->exceptionProg, i, p);
}
API RTresult rtContextSetMissProgram(void* h, unsigned i, void* p)
{
    GET_CTX("rtContextSetMissProgram");
    return setProgSlot(_cs, "rtContextSetMissProgram", c, c->miss, i, p);
}
API RTresult rtContextGetMissProgram(void* h, unsigned i, void** p)
{
    GET_CTX("rtContextGetMissProgram");
    return getProgSlot(_cs, "rtContextGetMissProgram", c, c->miss, i, p);
}

API RTresult rtContextSetExceptionEnabled(void* h, unsigned exception, int enabled)
{
    GET_CTX("rtContextSetExceptionEnabled");
    c->exceptionEnabled[exception] = enabled;
    return R(_cs, RT_SUCCESS, "rtContextSetExceptionEnabled(%s, 0x%x, %d)", N(c), exception, enabled);
}

API RTresult rtContextGetExceptionEnabled(void* h, unsigned exception, int* enabled)
{
    GET_CTX("rtContextGetExceptionEnabled");
    if (!enabled) return R(_cs, RT_ERROR_INVALID_VALUE, "rtContextGetExceptionEnabled(%s, 0x%x, NULL)", N(c), exception);
    auto it = c->exceptionEnabled.find(exception);
    *enabled = it == c->exceptionEnabled.end() ? 0 : it->second;
    return R(_cs, RT_SUCCESS, "rtContextGetExceptionEnabled(%s, 0x%x) = %d", N(c), exception, *enabled);
}

API RTresult rtContextCompile(void* h)
{
    GET_CTX("rtContextCompile");
    return R(_cs, RT_SUCCESS, "rtContextCompile(%s)", N(c));
}

static RTresult launch(const CallScope& cs, const char* fn, Context* c, unsigned entry, RTsize w, RTsize hgt, RTsize d)
{
    ++c->launches;
    if (c->dumpedStructVersion != g_structVersion) {
        c->dumpedStructVersion = g_structVersion;
        char reason[96];
        std::snprintf(reason, sizeof(reason), "before launch #%llu, entry %u, %ux%ux%u", c->launches, entry,
                      (unsigned)w, (unsigned)hgt, (unsigned)d);
        dumpScene(c, reason);
    }
    const bool rendered = bridgeLaunch(c);
    if (!rendered && g_testPattern) fillTestPattern(c);
    return R(cs, RT_SUCCESS, "%s(%s, entry %u, %ux%ux%u) [launch #%llu, %s]", fn, N(c), entry,
             (unsigned)w, (unsigned)hgt, (unsigned)d, c->launches, rendered ? "rendered by helper" : "not rendered");
}

API RTresult rtContextLaunch1D(void* h, unsigned entry, RTsize w)
{
    GET_CTX("rtContextLaunch1D");
    return launch(_cs, "rtContextLaunch1D", c, entry, w, 1, 1);
}
API RTresult rtContextLaunch2D(void* h, unsigned entry, RTsize w, RTsize hgt)
{
    GET_CTX("rtContextLaunch2D");
    return launch(_cs, "rtContextLaunch2D", c, entry, w, hgt, 1);
}
API RTresult rtContextLaunch3D(void* h, unsigned entry, RTsize w, RTsize hgt, RTsize d)
{
    GET_CTX("rtContextLaunch3D");
    return launch(_cs, "rtContextLaunch3D", c, entry, w, hgt, d);
}

API RTresult rtContextGetRunningState(void* h, int* running)
{
    GET_CTX("rtContextGetRunningState");
    if (!running) return R(_cs, RT_ERROR_INVALID_VALUE, "rtContextGetRunningState(%s, NULL)", N(c));
    *running = 0;
    return R(_cs, RT_SUCCESS, "rtContextGetRunningState(%s) = 0", N(c));
}

API RTresult rtContextSetPrintEnabled(void* h, int enabled)
{
    GET_CTX("rtContextSetPrintEnabled");
    c->printEnabled = enabled;
    return R(_cs, RT_SUCCESS, "rtContextSetPrintEnabled(%s, %d)", N(c), enabled);
}
API RTresult rtContextGetPrintEnabled(void* h, int* enabled)
{
    GET_CTX("rtContextGetPrintEnabled");
    if (!enabled) return R(_cs, RT_ERROR_INVALID_VALUE, "rtContextGetPrintEnabled(%s, NULL)", N(c));
    *enabled = c->printEnabled;
    return R(_cs, RT_SUCCESS, "rtContextGetPrintEnabled(%s) = %d", N(c), *enabled);
}
API RTresult rtContextSetPrintBufferSize(void* h, RTsize size)
{
    GET_CTX("rtContextSetPrintBufferSize");
    c->printBufferSize = size;
    return R(_cs, RT_SUCCESS, "rtContextSetPrintBufferSize(%s, %u)", N(c), (unsigned)size);
}
API RTresult rtContextGetPrintBufferSize(void* h, RTsize* size)
{
    GET_CTX("rtContextGetPrintBufferSize");
    if (!size) return R(_cs, RT_ERROR_INVALID_VALUE, "rtContextGetPrintBufferSize(%s, NULL)", N(c));
    *size = c->printBufferSize;
    return R(_cs, RT_SUCCESS, "rtContextGetPrintBufferSize(%s) = %u", N(c), (unsigned)*size);
}
API RTresult rtContextSetPrintLaunchIndex(void* h, int x, int y, int z)
{
    GET_CTX("rtContextSetPrintLaunchIndex");
    c->printIndex[0] = x; c->printIndex[1] = y; c->printIndex[2] = z;
    return R(_cs, RT_SUCCESS, "rtContextSetPrintLaunchIndex(%s, %d, %d, %d)", N(c), x, y, z);
}
API RTresult rtContextGetPrintLaunchIndex(void* h, int* x, int* y, int* z)
{
    GET_CTX("rtContextGetPrintLaunchIndex");
    if (x) *x = c->printIndex[0];
    if (y) *y = c->printIndex[1];
    if (z) *z = c->printIndex[2];
    return R(_cs, RT_SUCCESS, "rtContextGetPrintLaunchIndex(%s)", N(c));
}

// ===========================================================================
// API: Program
// ===========================================================================

static RTresult makeProgram(const CallScope& cs, const char* fn, Context* c, const std::string& ptx,
                            const char* name, const char* file, void** out)
{
    if (!name || !out) return R(cs, RT_ERROR_INVALID_VALUE, "%s(%s, name %s)", fn, N(c), name ? name : "NULL");
    const unsigned long long h = fnv1a(ptx.data(), ptx.size());
    if (g_ptxSeen.insert(h).second) {
        char path[64];
        std::snprintf(path, sizeof(path), "\\ptx\\%016llx.ptx", h);
        writeFile(g_dir + path, ptx.data(), ptx.size());
        logLine("  new PTX source %016llx (%u bytes) saved to ptx folder", h, (unsigned)ptx.size());
    }
    if (ptx.find(name) == std::string::npos)
        logLine("  WARNING: entry \"%s\" does not appear in PTX %016llx", name, h);
    Program* p = mk<Program>(K_PROGRAM, c);
    p->name = name;
    p->ptxHash = h;
    if (file) p->file = file;
    *out = p;
    return R(cs, RT_SUCCESS, "%s(%s, ptx %016llx, \"%s\"%s%s) = %s", fn, N(c), h, name,
             file ? ", file " : "", file ? file : "", N(p));
}

API RTresult rtProgramCreateFromPTXString(void* h, const char* ptx, const char* name, void** out)
{
    GET_CTX("rtProgramCreateFromPTXString");
    if (!ptx) return R(_cs, RT_ERROR_INVALID_VALUE, "rtProgramCreateFromPTXString(%s, NULL ptx)", N(c));
    return makeProgram(_cs, "rtProgramCreateFromPTXString", c, ptx, name, nullptr, out);
}

API RTresult rtProgramCreateFromPTXFile(void* h, const char* file, const char* name, void** out)
{
    GET_CTX("rtProgramCreateFromPTXFile");
    if (!file) return R(_cs, RT_ERROR_INVALID_VALUE, "rtProgramCreateFromPTXFile(%s, NULL file)", N(c));
    FILE* f = std::fopen(file, "rb");
    if (!f) return R(_cs, RT_ERROR_FILE_NOT_FOUND, "rtProgramCreateFromPTXFile(%s, \"%s\")", N(c), file);
    std::string ptx;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) ptx.append(buf, n);
    std::fclose(f);
    return makeProgram(_cs, "rtProgramCreateFromPTXFile", c, ptx, name, file, out);
}

API RTresult rtTextureSamplerGetId(void* h, int* id)
{
    ENTER("rtTextureSamplerGetId");
    TexSampler* t = as<TexSampler>(h, K_TEXSAMPLER);
    if (!t) return badHandle(_cs, "rtTextureSamplerGetId", K_TEXSAMPLER, h);
    if (!id) return R(_cs, RT_ERROR_INVALID_VALUE, "rtTextureSamplerGetId(%s, NULL)", N(t));
    *id = (int)t->id;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerGetId(%s) = %d", N(t), *id);
}

// ===========================================================================
// API: Selector, Transform
// ===========================================================================

API RTresult rtSelectorSetVisitProgram(void* h, void* hp)
{
    ENTER("rtSelectorSetVisitProgram");
    Selector* s = as<Selector>(h, K_SELECTOR);
    if (!s) return badHandle(_cs, "rtSelectorSetVisitProgram", K_SELECTOR, h);
    Program* p = as<Program>(hp, K_PROGRAM);
    if (!p) return badHandle(_cs, "rtSelectorSetVisitProgram", K_PROGRAM, hp);
    s->visit = p;
    return R(_cs, RT_SUCCESS, "rtSelectorSetVisitProgram(%s, %s \"%s\")", N(s), N(p), p->name.c_str());
}

API RTresult rtSelectorGetVisitProgram(void* h, void** out)
{
    ENTER("rtSelectorGetVisitProgram");
    Selector* s = as<Selector>(h, K_SELECTOR);
    if (!s) return badHandle(_cs, "rtSelectorGetVisitProgram", K_SELECTOR, h);
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "rtSelectorGetVisitProgram(%s, NULL)", N(s));
    *out = s->visit;
    return R(_cs, RT_SUCCESS, "rtSelectorGetVisitProgram(%s) = %s", N(s), N(s->visit));
}

API RTresult rtTransformSetMatrix(void* h, int transpose, const float* matrix, const float* inverse)
{
    ENTER("rtTransformSetMatrix");
    Transform* t = as<Transform>(h, K_TRANSFORM);
    if (!t) return badHandle(_cs, "rtTransformSetMatrix", K_TRANSFORM, h);
    if (!matrix) return R(_cs, RT_ERROR_INVALID_VALUE, "rtTransformSetMatrix(%s, %d, NULL)", N(t), transpose);
    // Stored row-major. transpose != 0 means the input is column-major.
    if (transpose) transpose4x4(matrix, t->m);
    else std::memcpy(t->m, matrix, sizeof(t->m));
    if (inverse) {
        if (transpose) transpose4x4(inverse, t->inv);
        else std::memcpy(t->inv, inverse, sizeof(t->inv));
    } else if (!invert4x4(t->m, t->inv)) {
        std::memset(t->inv, 0, sizeof(t->inv));
    }
    return R(_cs, RT_SUCCESS, "rtTransformSetMatrix(%s, transpose %d, %s%s)", N(t), transpose,
             fmtFloats(t->m, 16).c_str(), inverse ? ", with inverse" : "");
}

API RTresult rtTransformGetMatrix(void* h, int transpose, float* matrix, float* inverse)
{
    ENTER("rtTransformGetMatrix");
    Transform* t = as<Transform>(h, K_TRANSFORM);
    if (!t) return badHandle(_cs, "rtTransformGetMatrix", K_TRANSFORM, h);
    if (matrix) {
        if (transpose) transpose4x4(t->m, matrix);
        else std::memcpy(matrix, t->m, sizeof(t->m));
    }
    if (inverse) {
        if (transpose) transpose4x4(t->inv, inverse);
        else std::memcpy(inverse, t->inv, sizeof(t->inv));
    }
    return R(_cs, RT_SUCCESS, "rtTransformGetMatrix(%s, %d)", N(t), transpose);
}

API RTresult rtTransformSetChild(void* h, void* child)
{
    ENTER("rtTransformSetChild");
    Transform* t = as<Transform>(h, K_TRANSFORM);
    if (!t) return badHandle(_cs, "rtTransformSetChild", K_TRANSFORM, h);
    Obj* c = asAny(child);
    if (!c) return R(_cs, RT_ERROR_INVALID_VALUE, "rtTransformSetChild(%s, %s)", N(t), N(child));
    t->child = c;
    return R(_cs, RT_SUCCESS, "rtTransformSetChild(%s, %s)", N(t), N(c));
}

API RTresult rtTransformGetChild(void* h, void** out)
{
    ENTER("rtTransformGetChild");
    Transform* t = as<Transform>(h, K_TRANSFORM);
    if (!t) return badHandle(_cs, "rtTransformGetChild", K_TRANSFORM, h);
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "rtTransformGetChild(%s, NULL)", N(t));
    *out = t->child;
    return R(_cs, RT_SUCCESS, "rtTransformGetChild(%s) = %s", N(t), N(t->child));
}

API RTresult rtTransformGetChildType(void* h, unsigned* type)
{
    ENTER("rtTransformGetChildType");
    Transform* t = as<Transform>(h, K_TRANSFORM);
    if (!t) return badHandle(_cs, "rtTransformGetChildType", K_TRANSFORM, h);
    if (!type) return R(_cs, RT_ERROR_INVALID_VALUE, "rtTransformGetChildType(%s, NULL)", N(t));
    *type = objectTypeOf(t->child);
    return R(_cs, RT_SUCCESS, "rtTransformGetChildType(%s) = %s", N(t), typeName(*type));
}

// ===========================================================================
// API: Acceleration
// ===========================================================================

#define GET_ACCEL(fn)                                   \
    ENTER(fn);                                          \
    Accel* a = as<Accel>(h, K_ACCEL);                   \
    if (!a) return badHandle(_cs, fn, K_ACCEL, h)

API RTresult rtAccelerationSetBuilder(void* h, const char* s)
{
    GET_ACCEL("rtAccelerationSetBuilder");
    a->builder = s ? s : "";
    a->dirty = true;
    return R(_cs, RT_SUCCESS, "rtAccelerationSetBuilder(%s, \"%s\")", N(a), a->builder.c_str());
}
API RTresult rtAccelerationGetBuilder(void* h, const char** s)
{
    GET_ACCEL("rtAccelerationGetBuilder");
    if (s) *s = a->builder.c_str();
    return R(_cs, RT_SUCCESS, "rtAccelerationGetBuilder(%s) = \"%s\"", N(a), a->builder.c_str());
}
API RTresult rtAccelerationSetTraverser(void* h, const char* s)
{
    GET_ACCEL("rtAccelerationSetTraverser");
    a->traverser = s ? s : "";
    return R(_cs, RT_SUCCESS, "rtAccelerationSetTraverser(%s, \"%s\")", N(a), a->traverser.c_str());
}
API RTresult rtAccelerationGetTraverser(void* h, const char** s)
{
    GET_ACCEL("rtAccelerationGetTraverser");
    if (s) *s = a->traverser.c_str();
    return R(_cs, RT_SUCCESS, "rtAccelerationGetTraverser(%s) = \"%s\"", N(a), a->traverser.c_str());
}
API RTresult rtAccelerationSetProperty(void* h, const char* name, const char* value)
{
    GET_ACCEL("rtAccelerationSetProperty");
    if (!name || !value) return R(_cs, RT_ERROR_INVALID_VALUE, "rtAccelerationSetProperty(%s, NULL)", N(a));
    a->props[name] = value;
    return R(_cs, RT_SUCCESS, "rtAccelerationSetProperty(%s, \"%s\", \"%s\")", N(a), name, value);
}
API RTresult rtAccelerationGetProperty(void* h, const char* name, const char** value)
{
    GET_ACCEL("rtAccelerationGetProperty");
    if (!name || !value) return R(_cs, RT_ERROR_INVALID_VALUE, "rtAccelerationGetProperty(%s, NULL)", N(a));
    static const std::string empty;
    auto it = a->props.find(name);
    *value = it == a->props.end() ? empty.c_str() : it->second.c_str();
    return R(_cs, RT_SUCCESS, "rtAccelerationGetProperty(%s, \"%s\") = \"%s\"", N(a), name, *value);
}

// Acceleration caching: the demo saves "acceleration data" to its cache folder
// and restores it later. We hand out a small marker blob and accept any data.
static const char kAccelBlob[16] = "DGBRIDGE-ACCEL1";

API RTresult rtAccelerationGetDataSize(void* h, RTsize* size)
{
    GET_ACCEL("rtAccelerationGetDataSize");
    if (!size) return R(_cs, RT_ERROR_INVALID_VALUE, "rtAccelerationGetDataSize(%s, NULL)", N(a));
    *size = sizeof(kAccelBlob);
    return R(_cs, RT_SUCCESS, "rtAccelerationGetDataSize(%s) = %u", N(a), (unsigned)*size);
}
API RTresult rtAccelerationGetData(void* h, void* data)
{
    GET_ACCEL("rtAccelerationGetData");
    if (!data) return R(_cs, RT_ERROR_INVALID_VALUE, "rtAccelerationGetData(%s, NULL)", N(a));
    std::memcpy(data, kAccelBlob, sizeof(kAccelBlob));
    return R(_cs, RT_SUCCESS, "rtAccelerationGetData(%s)", N(a));
}
API RTresult rtAccelerationSetData(void* h, const void* data, RTsize size)
{
    GET_ACCEL("rtAccelerationSetData");
    a->dirty = false;
    return R(_cs, RT_SUCCESS, "rtAccelerationSetData(%s, %u bytes: %s)", N(a), (unsigned)size,
             data ? fmtHex(data, size, 16).c_str() : "NULL");
}
API RTresult rtAccelerationMarkDirty(void* h)
{
    GET_ACCEL("rtAccelerationMarkDirty");
    a->dirty = true;
    return R(_cs, RT_SUCCESS, "rtAccelerationMarkDirty(%s)", N(a));
}
API RTresult rtAccelerationIsDirty(void* h, int* dirty)
{
    GET_ACCEL("rtAccelerationIsDirty");
    if (!dirty) return R(_cs, RT_ERROR_INVALID_VALUE, "rtAccelerationIsDirty(%s, NULL)", N(a));
    *dirty = a->dirty ? 1 : 0;
    return R(_cs, RT_SUCCESS, "rtAccelerationIsDirty(%s) = %d", N(a), *dirty);
}

// ===========================================================================
// API: GeometryInstance, Geometry, Material
// ===========================================================================

#define GET_GI(fn)                                      \
    ENTER(fn);                                          \
    GInst* g = as<GInst>(h, K_GINST);                   \
    if (!g) return badHandle(_cs, fn, K_GINST, h)

API RTresult rtGeometryInstanceSetGeometry(void* h, void* hg)
{
    GET_GI("rtGeometryInstanceSetGeometry");
    Geometry* geo = as<Geometry>(hg, K_GEOMETRY);
    if (!geo) return badHandle(_cs, "rtGeometryInstanceSetGeometry", K_GEOMETRY, hg);
    g->geometry = geo;
    return R(_cs, RT_SUCCESS, "rtGeometryInstanceSetGeometry(%s, %s)", N(g), N(geo));
}
API RTresult rtGeometryInstanceGetGeometry(void* h, void** out)
{
    GET_GI("rtGeometryInstanceGetGeometry");
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "rtGeometryInstanceGetGeometry(%s, NULL)", N(g));
    *out = g->geometry;
    return R(_cs, RT_SUCCESS, "rtGeometryInstanceGetGeometry(%s) = %s", N(g), N(g->geometry));
}
API RTresult rtGeometryInstanceSetMaterialCount(void* h, unsigned n)
{
    GET_GI("rtGeometryInstanceSetMaterialCount");
    g->materials.resize(n, nullptr);
    return R(_cs, RT_SUCCESS, "rtGeometryInstanceSetMaterialCount(%s, %u)", N(g), n);
}
API RTresult rtGeometryInstanceGetMaterialCount(void* h, unsigned* n)
{
    GET_GI("rtGeometryInstanceGetMaterialCount");
    if (!n) return R(_cs, RT_ERROR_INVALID_VALUE, "rtGeometryInstanceGetMaterialCount(%s, NULL)", N(g));
    *n = (unsigned)g->materials.size();
    return R(_cs, RT_SUCCESS, "rtGeometryInstanceGetMaterialCount(%s) = %u", N(g), *n);
}
API RTresult rtGeometryInstanceSetMaterial(void* h, unsigned i, void* hm)
{
    GET_GI("rtGeometryInstanceSetMaterial");
    Material* m = as<Material>(hm, K_MATERIAL);
    if (!m || i >= g->materials.size())
        return R(_cs, RT_ERROR_INVALID_VALUE, "rtGeometryInstanceSetMaterial(%s, %u, %s) [count %u]", N(g), i, N(hm),
                 (unsigned)g->materials.size());
    g->materials[i] = m;
    return R(_cs, RT_SUCCESS, "rtGeometryInstanceSetMaterial(%s, %u, %s)", N(g), i, N(m));
}
API RTresult rtGeometryInstanceGetMaterial(void* h, unsigned i, void** out)
{
    GET_GI("rtGeometryInstanceGetMaterial");
    if (!out || i >= g->materials.size()) return R(_cs, RT_ERROR_INVALID_VALUE, "rtGeometryInstanceGetMaterial(%s, %u)", N(g), i);
    *out = g->materials[i];
    return R(_cs, RT_SUCCESS, "rtGeometryInstanceGetMaterial(%s, %u) = %s", N(g), i, N(*out));
}

#define GET_GEO(fn)                                     \
    ENTER(fn);                                          \
    Geometry* g = as<Geometry>(h, K_GEOMETRY);          \
    if (!g) return badHandle(_cs, fn, K_GEOMETRY, h)

API RTresult rtGeometrySetPrimitiveCount(void* h, unsigned n)
{
    GET_GEO("rtGeometrySetPrimitiveCount");
    g->primCount = n;
    return R(_cs, RT_SUCCESS, "rtGeometrySetPrimitiveCount(%s, %u)", N(g), n);
}
API RTresult rtGeometryGetPrimitiveCount(void* h, unsigned* n)
{
    GET_GEO("rtGeometryGetPrimitiveCount");
    if (!n) return R(_cs, RT_ERROR_INVALID_VALUE, "rtGeometryGetPrimitiveCount(%s, NULL)", N(g));
    *n = g->primCount;
    return R(_cs, RT_SUCCESS, "rtGeometryGetPrimitiveCount(%s) = %u", N(g), *n);
}
API RTresult rtGeometrySetBoundingBoxProgram(void* h, void* hp)
{
    GET_GEO("rtGeometrySetBoundingBoxProgram");
    Program* p = as<Program>(hp, K_PROGRAM);
    if (!p) return badHandle(_cs, "rtGeometrySetBoundingBoxProgram", K_PROGRAM, hp);
    g->bbox = p;
    return R(_cs, RT_SUCCESS, "rtGeometrySetBoundingBoxProgram(%s, %s \"%s\")", N(g), N(p), p->name.c_str());
}
API RTresult rtGeometryGetBoundingBoxProgram(void* h, void** out)
{
    GET_GEO("rtGeometryGetBoundingBoxProgram");
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "rtGeometryGetBoundingBoxProgram(%s, NULL)", N(g));
    *out = g->bbox;
    return R(_cs, RT_SUCCESS, "rtGeometryGetBoundingBoxProgram(%s) = %s", N(g), N(g->bbox));
}
API RTresult rtGeometrySetIntersectionProgram(void* h, void* hp)
{
    GET_GEO("rtGeometrySetIntersectionProgram");
    Program* p = as<Program>(hp, K_PROGRAM);
    if (!p) return badHandle(_cs, "rtGeometrySetIntersectionProgram", K_PROGRAM, hp);
    g->isect = p;
    return R(_cs, RT_SUCCESS, "rtGeometrySetIntersectionProgram(%s, %s \"%s\")", N(g), N(p), p->name.c_str());
}
API RTresult rtGeometryGetIntersectionProgram(void* h, void** out)
{
    GET_GEO("rtGeometryGetIntersectionProgram");
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "rtGeometryGetIntersectionProgram(%s, NULL)", N(g));
    *out = g->isect;
    return R(_cs, RT_SUCCESS, "rtGeometryGetIntersectionProgram(%s) = %s", N(g), N(g->isect));
}
API RTresult rtGeometryMarkDirty(void* h)
{
    GET_GEO("rtGeometryMarkDirty");
    g->dirty = true;
    return R(_cs, RT_SUCCESS, "rtGeometryMarkDirty(%s)", N(g));
}
API RTresult rtGeometryIsDirty(void* h, int* dirty)
{
    GET_GEO("rtGeometryIsDirty");
    if (!dirty) return R(_cs, RT_ERROR_INVALID_VALUE, "rtGeometryIsDirty(%s, NULL)", N(g));
    *dirty = g->dirty ? 1 : 0;
    return R(_cs, RT_SUCCESS, "rtGeometryIsDirty(%s) = %d", N(g), *dirty);
}

static RTresult materialSet(const char* fn, void* h, unsigned rayType, void* hp, bool closest)
{
    ENTER(fn);
    Material* m = as<Material>(h, K_MATERIAL);
    if (!m) return badHandle(_cs, fn, K_MATERIAL, h);
    Program* p = as<Program>(hp, K_PROGRAM);
    if (!p) return badHandle(_cs, fn, K_PROGRAM, hp);
    (closest ? m->ch : m->ah)[rayType] = p;
    return R(_cs, RT_SUCCESS, "%s(%s, rayType %u, %s \"%s\")", fn, N(m), rayType, N(p), p->name.c_str());
}

static RTresult materialGet(const char* fn, void* h, unsigned rayType, void** out, bool closest)
{
    ENTER(fn);
    Material* m = as<Material>(h, K_MATERIAL);
    if (!m) return badHandle(_cs, fn, K_MATERIAL, h);
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "%s(%s, %u, NULL)", fn, N(m), rayType);
    auto& slots = closest ? m->ch : m->ah;
    auto it = slots.find(rayType);
    *out = it == slots.end() ? nullptr : it->second;
    return R(_cs, RT_SUCCESS, "%s(%s, rayType %u) = %s", fn, N(m), rayType, N(*out));
}

API RTresult rtMaterialSetClosestHitProgram(void* h, unsigned r, void* p) { return materialSet("rtMaterialSetClosestHitProgram", h, r, p, true); }
API RTresult rtMaterialGetClosestHitProgram(void* h, unsigned r, void** p) { return materialGet("rtMaterialGetClosestHitProgram", h, r, p, true); }
API RTresult rtMaterialSetAnyHitProgram(void* h, unsigned r, void* p) { return materialSet("rtMaterialSetAnyHitProgram", h, r, p, false); }
API RTresult rtMaterialGetAnyHitProgram(void* h, unsigned r, void** p) { return materialGet("rtMaterialGetAnyHitProgram", h, r, p, false); }

// ===========================================================================
// API: TextureSampler
// ===========================================================================

#define GET_TEX(fn)                                     \
    ENTER(fn);                                          \
    TexSampler* t = as<TexSampler>(h, K_TEXSAMPLER);    \
    if (!t) return badHandle(_cs, fn, K_TEXSAMPLER, h)

API RTresult rtTextureSamplerSetMipLevelCount(void* h, unsigned n)
{
    GET_TEX("rtTextureSamplerSetMipLevelCount");
    t->mipCount = n;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerSetMipLevelCount(%s, %u)", N(t), n);
}
API RTresult rtTextureSamplerGetMipLevelCount(void* h, unsigned* n)
{
    GET_TEX("rtTextureSamplerGetMipLevelCount");
    if (n) *n = t->mipCount;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerGetMipLevelCount(%s) = %u", N(t), t->mipCount);
}
API RTresult rtTextureSamplerSetArraySize(void* h, unsigned n)
{
    GET_TEX("rtTextureSamplerSetArraySize");
    t->arraySize = n;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerSetArraySize(%s, %u)", N(t), n);
}
API RTresult rtTextureSamplerGetArraySize(void* h, unsigned* n)
{
    GET_TEX("rtTextureSamplerGetArraySize");
    if (n) *n = t->arraySize;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerGetArraySize(%s) = %u", N(t), t->arraySize);
}
API RTresult rtTextureSamplerSetWrapMode(void* h, unsigned dim, unsigned mode)
{
    GET_TEX("rtTextureSamplerSetWrapMode");
    if (dim < 3) t->wrap[dim] = mode;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerSetWrapMode(%s, dim %u, %u)", N(t), dim, mode);
}
API RTresult rtTextureSamplerGetWrapMode(void* h, unsigned dim, unsigned* mode)
{
    GET_TEX("rtTextureSamplerGetWrapMode");
    if (mode) *mode = dim < 3 ? t->wrap[dim] : 0;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerGetWrapMode(%s, dim %u)", N(t), dim);
}
API RTresult rtTextureSamplerSetFilteringModes(void* h, unsigned minF, unsigned magF, unsigned mipF)
{
    GET_TEX("rtTextureSamplerSetFilteringModes");
    t->filter[0] = minF; t->filter[1] = magF; t->filter[2] = mipF;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerSetFilteringModes(%s, %u, %u, %u)", N(t), minF, magF, mipF);
}
API RTresult rtTextureSamplerGetFilteringModes(void* h, unsigned* minF, unsigned* magF, unsigned* mipF)
{
    GET_TEX("rtTextureSamplerGetFilteringModes");
    if (minF) *minF = t->filter[0];
    if (magF) *magF = t->filter[1];
    if (mipF) *mipF = t->filter[2];
    return R(_cs, RT_SUCCESS, "rtTextureSamplerGetFilteringModes(%s)", N(t));
}
API RTresult rtTextureSamplerSetMaxAnisotropy(void* h, float v)
{
    GET_TEX("rtTextureSamplerSetMaxAnisotropy");
    t->aniso = v;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerSetMaxAnisotropy(%s, %g)", N(t), v);
}
API RTresult rtTextureSamplerGetMaxAnisotropy(void* h, float* v)
{
    GET_TEX("rtTextureSamplerGetMaxAnisotropy");
    if (v) *v = t->aniso;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerGetMaxAnisotropy(%s) = %g", N(t), t->aniso);
}
API RTresult rtTextureSamplerSetReadMode(void* h, unsigned m)
{
    GET_TEX("rtTextureSamplerSetReadMode");
    t->readMode = m;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerSetReadMode(%s, %u)", N(t), m);
}
API RTresult rtTextureSamplerGetReadMode(void* h, unsigned* m)
{
    GET_TEX("rtTextureSamplerGetReadMode");
    if (m) *m = t->readMode;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerGetReadMode(%s) = %u", N(t), t->readMode);
}
API RTresult rtTextureSamplerSetIndexingMode(void* h, unsigned m)
{
    GET_TEX("rtTextureSamplerSetIndexingMode");
    t->indexMode = m;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerSetIndexingMode(%s, %u)", N(t), m);
}
API RTresult rtTextureSamplerGetIndexingMode(void* h, unsigned* m)
{
    GET_TEX("rtTextureSamplerGetIndexingMode");
    if (m) *m = t->indexMode;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerGetIndexingMode(%s) = %u", N(t), t->indexMode);
}
API RTresult rtTextureSamplerSetBuffer(void* h, unsigned arrayIdx, unsigned mip, void* hb)
{
    GET_TEX("rtTextureSamplerSetBuffer");
    Buffer* b = as<Buffer>(hb, K_BUFFER);
    if (!b) return badHandle(_cs, "rtTextureSamplerSetBuffer", K_BUFFER, hb);
    t->buffers[{ arrayIdx, mip }] = b;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerSetBuffer(%s, %u, %u, %s)", N(t), arrayIdx, mip, N(b));
}
API RTresult rtTextureSamplerGetBuffer(void* h, unsigned arrayIdx, unsigned mip, void** out)
{
    GET_TEX("rtTextureSamplerGetBuffer");
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "rtTextureSamplerGetBuffer(%s, NULL)", N(t));
    auto it = t->buffers.find({ arrayIdx, mip });
    *out = it == t->buffers.end() ? nullptr : it->second;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerGetBuffer(%s, %u, %u) = %s", N(t), arrayIdx, mip, N(*out));
}
API RTresult rtTextureSamplerCreateFromGLImage(void* h, unsigned glId, unsigned target, void** out)
{
    GET_CTX("rtTextureSamplerCreateFromGLImage");
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "rtTextureSamplerCreateFromGLImage(%s, NULL)", N(c));
    TexSampler* t = mk<TexSampler>(K_TEXSAMPLER, c);
    t->glId = glId;
    t->glTarget = target;
    *out = t;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerCreateFromGLImage(%s, gl %u, target 0x%x) = %s", N(c), glId, target, N(t));
}
API RTresult rtTextureSamplerGetGLImageId(void* h, unsigned* id)
{
    GET_TEX("rtTextureSamplerGetGLImageId");
    if (id) *id = t->glId;
    return R(_cs, RT_SUCCESS, "rtTextureSamplerGetGLImageId(%s) = %u", N(t), t->glId);
}
API RTresult rtTextureSamplerGLRegister(void* h)
{
    GET_TEX("rtTextureSamplerGLRegister");
    return R(_cs, RT_SUCCESS, "rtTextureSamplerGLRegister(%s)", N(t));
}
API RTresult rtTextureSamplerGLUnregister(void* h)
{
    GET_TEX("rtTextureSamplerGLUnregister");
    return R(_cs, RT_SUCCESS, "rtTextureSamplerGLUnregister(%s)", N(t));
}

// ===========================================================================
// API: Buffer
// ===========================================================================

#define GET_BUF(fn)                                     \
    ENTER(fn);                                          \
    Buffer* b = as<Buffer>(h, K_BUFFER);                \
    if (!b) return badHandle(_cs, fn, K_BUFFER, h)

API RTresult rtBufferCreate(void* h, unsigned desc, void** out)
{
    GET_CTX("rtBufferCreate");
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "rtBufferCreate(%s, 0x%x, NULL)", N(c), desc);
    Buffer* b = mk<Buffer>(K_BUFFER, c);
    b->desc = desc;
    *out = b;
    return R(_cs, RT_SUCCESS, "rtBufferCreate(%s, desc 0x%x) = %s", N(c), desc, N(b));
}

API RTresult rtBufferCreateFromGLBO(void* h, unsigned desc, unsigned glId, void** out)
{
    GET_CTX("rtBufferCreateFromGLBO");
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "rtBufferCreateFromGLBO(%s, NULL)", N(c));
    Buffer* b = mk<Buffer>(K_BUFFER, c);
    b->desc = desc;
    b->glId = glId;
    *out = b;
    return R(_cs, RT_SUCCESS, "rtBufferCreateFromGLBO(%s, desc 0x%x, gl %u) = %s", N(c), desc, glId, N(b));
}

API RTresult rtBufferSetFormat(void* h, unsigned fmt)
{
    GET_BUF("rtBufferSetFormat");
    b->format = fmt;
    return R(_cs, RT_SUCCESS, "rtBufferSetFormat(%s, %s)", N(b), formatName(fmt).c_str());
}
API RTresult rtBufferGetFormat(void* h, unsigned* fmt)
{
    GET_BUF("rtBufferGetFormat");
    if (!fmt) return R(_cs, RT_ERROR_INVALID_VALUE, "rtBufferGetFormat(%s, NULL)", N(b));
    *fmt = b->format;
    return R(_cs, RT_SUCCESS, "rtBufferGetFormat(%s) = %s", N(b), formatName(b->format).c_str());
}
API RTresult rtBufferSetElementSize(void* h, RTsize size)
{
    GET_BUF("rtBufferSetElementSize");
    b->elemSize = size;
    return R(_cs, RT_SUCCESS, "rtBufferSetElementSize(%s, %u)", N(b), (unsigned)size);
}
API RTresult rtBufferGetElementSize(void* h, RTsize* size)
{
    GET_BUF("rtBufferGetElementSize");
    if (!size) return R(_cs, RT_ERROR_INVALID_VALUE, "rtBufferGetElementSize(%s, NULL)", N(b));
    *size = b->elementBytes();
    return R(_cs, RT_SUCCESS, "rtBufferGetElementSize(%s) = %u", N(b), (unsigned)*size);
}

static RTresult setSize(const CallScope& cs, const char* fn, Buffer* b, unsigned dimensionality, RTsize w, RTsize h, RTsize d)
{
    b->dimensionality = dimensionality;
    b->dims[0] = w;
    b->dims[1] = dimensionality >= 2 ? h : 1;
    b->dims[2] = dimensionality >= 3 ? d : 1;
    return R(cs, RT_SUCCESS, "%s(%s, %ux%ux%u) [%s, %u bytes]", fn, N(b), (unsigned)b->dims[0], (unsigned)b->dims[1],
             (unsigned)b->dims[2], formatName(b->format).c_str(), (unsigned)b->totalBytes());
}

API RTresult rtBufferSetSize1D(void* h, RTsize w)
{
    GET_BUF("rtBufferSetSize1D");
    return setSize(_cs, "rtBufferSetSize1D", b, 1, w, 1, 1);
}
API RTresult rtBufferSetSize2D(void* h, RTsize w, RTsize hh)
{
    GET_BUF("rtBufferSetSize2D");
    return setSize(_cs, "rtBufferSetSize2D", b, 2, w, hh, 1);
}
API RTresult rtBufferSetSize3D(void* h, RTsize w, RTsize hh, RTsize d)
{
    GET_BUF("rtBufferSetSize3D");
    return setSize(_cs, "rtBufferSetSize3D", b, 3, w, hh, d);
}
API RTresult rtBufferSetSizev(void* h, unsigned dimensionality, const RTsize* dims)
{
    GET_BUF("rtBufferSetSizev");
    if (!dims || dimensionality < 1 || dimensionality > 3)
        return R(_cs, RT_ERROR_INVALID_VALUE, "rtBufferSetSizev(%s, %u)", N(b), dimensionality);
    return setSize(_cs, "rtBufferSetSizev", b, dimensionality, dims[0],
                   dimensionality >= 2 ? dims[1] : 1, dimensionality >= 3 ? dims[2] : 1);
}
API RTresult rtBufferGetSize1D(void* h, RTsize* w)
{
    GET_BUF("rtBufferGetSize1D");
    if (w) *w = b->dims[0];
    return R(_cs, RT_SUCCESS, "rtBufferGetSize1D(%s) = %u", N(b), (unsigned)b->dims[0]);
}
API RTresult rtBufferGetSize2D(void* h, RTsize* w, RTsize* hh)
{
    GET_BUF("rtBufferGetSize2D");
    if (w) *w = b->dims[0];
    if (hh) *hh = b->dims[1];
    return R(_cs, RT_SUCCESS, "rtBufferGetSize2D(%s) = %ux%u", N(b), (unsigned)b->dims[0], (unsigned)b->dims[1]);
}
API RTresult rtBufferGetSize3D(void* h, RTsize* w, RTsize* hh, RTsize* d)
{
    GET_BUF("rtBufferGetSize3D");
    if (w) *w = b->dims[0];
    if (hh) *hh = b->dims[1];
    if (d) *d = b->dims[2];
    return R(_cs, RT_SUCCESS, "rtBufferGetSize3D(%s) = %ux%ux%u", N(b), (unsigned)b->dims[0], (unsigned)b->dims[1],
             (unsigned)b->dims[2]);
}
API RTresult rtBufferGetSizev(void* h, unsigned dimensionality, RTsize* dims)
{
    GET_BUF("rtBufferGetSizev");
    if (!dims) return R(_cs, RT_ERROR_INVALID_VALUE, "rtBufferGetSizev(%s, %u, NULL)", N(b), dimensionality);
    for (unsigned i = 0; i < dimensionality && i < 3; ++i) dims[i] = b->dims[i];
    return R(_cs, RT_SUCCESS, "rtBufferGetSizev(%s, %u)", N(b), dimensionality);
}
API RTresult rtBufferGetDimensionality(void* h, unsigned* dim)
{
    GET_BUF("rtBufferGetDimensionality");
    if (!dim) return R(_cs, RT_ERROR_INVALID_VALUE, "rtBufferGetDimensionality(%s, NULL)", N(b));
    *dim = b->dimensionality;
    return R(_cs, RT_SUCCESS, "rtBufferGetDimensionality(%s) = %u", N(b), *dim);
}

API RTresult rtBufferMap(void* h, void** p)
{
    GET_BUF("rtBufferMap");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtBufferMap(%s, NULL)", N(b));
    if (b->mapped) return R(_cs, RT_ERROR_ALREADY_MAPPED, "rtBufferMap(%s)", N(b));
    size_t need = b->totalBytes();
    if (need == 0) need = 16;
    if (b->host.size() < need) b->host.resize(need, 0);
    b->mapped = true;
    *p = b->host.data();
    return R(_cs, RT_SUCCESS, "rtBufferMap(%s) [%u bytes]", N(b), (unsigned)need);
}

API RTresult rtBufferUnmap(void* h)
{
    GET_BUF("rtBufferUnmap");
    if (!b->mapped) return R(_cs, RT_ERROR_INVALID_VALUE, "rtBufferUnmap(%s) [not mapped]", N(b));
    b->mapped = false;
    RTresult r = R(_cs, RT_SUCCESS, "rtBufferUnmap(%s)", N(b));
    if (!(b->desc & BUF_OUTPUT) || !b->glId) maybeDumpBuffer(b);
    return r;
}

API RTresult rtBufferGetGLBOId(void* h, unsigned* id)
{
    GET_BUF("rtBufferGetGLBOId");
    if (!id) return R(_cs, RT_ERROR_INVALID_VALUE, "rtBufferGetGLBOId(%s, NULL)", N(b));
    *id = b->glId;
    return R(_cs, RT_SUCCESS, "rtBufferGetGLBOId(%s) = %u", N(b), b->glId);
}
API RTresult rtBufferGLRegister(void* h)
{
    GET_BUF("rtBufferGLRegister");
    return R(_cs, RT_SUCCESS, "rtBufferGLRegister(%s)", N(b));
}
API RTresult rtBufferGLUnregister(void* h)
{
    GET_BUF("rtBufferGLUnregister");
    return R(_cs, RT_SUCCESS, "rtBufferGLUnregister(%s)", N(b));
}

// ===========================================================================
// API: Variable (non-generated part)
// ===========================================================================

#define GET_VAR(fn)                                     \
    ENTER(fn);                                          \
    Var* v = as<Var>(h, K_VARIABLE);                    \
    if (!v) return badHandle(_cs, fn, K_VARIABLE, h)

static RTresult varSet(const CallScope& cs, const char* fn, Var* v, unsigned type, const void* data, size_t n,
                       const std::string& shown)
{
    if (v->type != OT_UNKNOWN && v->type != type)
        logLine("  note: %s \"%s\" changes type %s -> %s", N(v), v->name.c_str(), typeName(v->type), typeName(type));
    if (v->owner && v->owner->kind == K_MATERIAL) ++g_materialVersion;
    v->type = type;
    v->object = nullptr;
    v->data.assign((const unsigned char*)data, (const unsigned char*)data + n);
    return R(cs, RT_SUCCESS, "%s(%s \"%s\" on %s, %s)", fn, N(v), v->name.c_str(), N(v->owner), shown.c_str());
}

static RTresult varGet(const CallScope& cs, const char* fn, Var* v, void* out, size_t n)
{
    std::memset(out, 0, n);
    std::memcpy(out, v->data.data(), std::min(n, v->data.size()));
    return R(cs, RT_SUCCESS, "%s(%s \"%s\") = %s", fn, N(v), v->name.c_str(), fmtVar(v).c_str());
}

API RTresult rtVariableSetObject(void* h, void* obj)
{
    GET_VAR("rtVariableSetObject");
    Obj* o = asAny(obj);
    if (!o) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSetObject(%s \"%s\", %s)", N(v), v->name.c_str(), N(obj));
    if (v->owner && v->owner->kind == K_MATERIAL) ++g_materialVersion;
    v->type = objectTypeOf(o);
    v->object = o;
    v->data.clear();
    return R(_cs, RT_SUCCESS, "rtVariableSetObject(%s \"%s\" on %s, %s)", N(v), v->name.c_str(), N(v->owner), N(o));
}

API RTresult rtVariableGetObject(void* h, void** out)
{
    GET_VAR("rtVariableGetObject");
    if (!out) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetObject(%s, NULL)", N(v));
    *out = v->object;
    return R(_cs, RT_SUCCESS, "rtVariableGetObject(%s \"%s\") = %s", N(v), v->name.c_str(), N(v->object));
}

API RTresult rtVariableSetUserData(void* h, RTsize size, const void* p)
{
    GET_VAR("rtVariableSetUserData");
    if (!p && size) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSetUserData(%s, %u, NULL)", N(v), (unsigned)size);
    char b[32];
    std::snprintf(b, sizeof(b), "%u bytes ", (unsigned)size);
    return varSet(_cs, "rtVariableSetUserData", v, OT_USER, p, size, b + fmtHex(p, size, 64));
}

API RTresult rtVariableGetUserData(void* h, RTsize size, void* p)
{
    GET_VAR("rtVariableGetUserData");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetUserData(%s, NULL)", N(v));
    return varGet(_cs, "rtVariableGetUserData", v, p, size);
}

API RTresult rtVariableGetName(void* h, const char** name)
{
    GET_VAR("rtVariableGetName");
    if (!name) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetName(%s, NULL)", N(v));
    *name = v->name.c_str();
    return R(_cs, RT_SUCCESS, "rtVariableGetName(%s) = \"%s\"", N(v), v->name.c_str());
}

API RTresult rtVariableGetAnnotation(void* h, const char** a)
{
    GET_VAR("rtVariableGetAnnotation");
    static const char empty[] = "";
    if (a) *a = empty;
    return R(_cs, RT_SUCCESS, "rtVariableGetAnnotation(%s \"%s\") = \"\"", N(v), v->name.c_str());
}

API RTresult rtVariableGetType(void* h, unsigned* type)
{
    GET_VAR("rtVariableGetType");
    if (!type) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetType(%s, NULL)", N(v));
    *type = v->type;
    return R(_cs, RT_SUCCESS, "rtVariableGetType(%s \"%s\") = %s", N(v), v->name.c_str(), typeName(v->type));
}

API RTresult rtVariableGetSize(void* h, RTsize* size)
{
    GET_VAR("rtVariableGetSize");
    if (!size) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetSize(%s, NULL)", N(v));
    *size = v->object ? sizeof(void*) : v->data.size();
    return R(_cs, RT_SUCCESS, "rtVariableGetSize(%s \"%s\") = %u", N(v), v->name.c_str(), (unsigned)*size);
}


// ===========================================================================
// API: Variable setters/getters (generated)
// ===========================================================================
API RTresult rtVariableSet1f(void* h, float a0)
{
    GET_VAR("rtVariableSet1f");
    const float v_[1] = { a0 };
    return varSet(_cs, "rtVariableSet1f", v, OT_FLOAT, v_, sizeof(v_), fmtFloats(v_, 1));
}
API RTresult rtVariableSet1fv(void* h, const float* p)
{
    GET_VAR("rtVariableSet1fv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSet1fv(%s, NULL)", N(v));
    return varSet(_cs, "rtVariableSet1fv", v, OT_FLOAT, p, 1 * sizeof(float), fmtFloats(p, 1));
}
API RTresult rtVariableGet1f(void* h, float* o0)
{
    GET_VAR("rtVariableGet1f");
    float v_[1];
    RTresult r_ = varGet(_cs, "rtVariableGet1f", v, v_, sizeof(v_));
    if (o0) *o0 = v_[0];
    return r_;
}
API RTresult rtVariableGet1fv(void* h, float* p)
{
    GET_VAR("rtVariableGet1fv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGet1fv(%s, NULL)", N(v));
    return varGet(_cs, "rtVariableGet1fv", v, p, 1 * sizeof(float));
}
API RTresult rtVariableSet2f(void* h, float a0, float a1)
{
    GET_VAR("rtVariableSet2f");
    const float v_[2] = { a0, a1 };
    return varSet(_cs, "rtVariableSet2f", v, OT_FLOAT2, v_, sizeof(v_), fmtFloats(v_, 2));
}
API RTresult rtVariableSet2fv(void* h, const float* p)
{
    GET_VAR("rtVariableSet2fv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSet2fv(%s, NULL)", N(v));
    return varSet(_cs, "rtVariableSet2fv", v, OT_FLOAT2, p, 2 * sizeof(float), fmtFloats(p, 2));
}
API RTresult rtVariableGet2f(void* h, float* o0, float* o1)
{
    GET_VAR("rtVariableGet2f");
    float v_[2];
    RTresult r_ = varGet(_cs, "rtVariableGet2f", v, v_, sizeof(v_));
    if (o0) *o0 = v_[0];
    if (o1) *o1 = v_[1];
    return r_;
}
API RTresult rtVariableGet2fv(void* h, float* p)
{
    GET_VAR("rtVariableGet2fv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGet2fv(%s, NULL)", N(v));
    return varGet(_cs, "rtVariableGet2fv", v, p, 2 * sizeof(float));
}
API RTresult rtVariableSet3f(void* h, float a0, float a1, float a2)
{
    GET_VAR("rtVariableSet3f");
    const float v_[3] = { a0, a1, a2 };
    return varSet(_cs, "rtVariableSet3f", v, OT_FLOAT3, v_, sizeof(v_), fmtFloats(v_, 3));
}
API RTresult rtVariableSet3fv(void* h, const float* p)
{
    GET_VAR("rtVariableSet3fv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSet3fv(%s, NULL)", N(v));
    return varSet(_cs, "rtVariableSet3fv", v, OT_FLOAT3, p, 3 * sizeof(float), fmtFloats(p, 3));
}
API RTresult rtVariableGet3f(void* h, float* o0, float* o1, float* o2)
{
    GET_VAR("rtVariableGet3f");
    float v_[3];
    RTresult r_ = varGet(_cs, "rtVariableGet3f", v, v_, sizeof(v_));
    if (o0) *o0 = v_[0];
    if (o1) *o1 = v_[1];
    if (o2) *o2 = v_[2];
    return r_;
}
API RTresult rtVariableGet3fv(void* h, float* p)
{
    GET_VAR("rtVariableGet3fv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGet3fv(%s, NULL)", N(v));
    return varGet(_cs, "rtVariableGet3fv", v, p, 3 * sizeof(float));
}
API RTresult rtVariableSet4f(void* h, float a0, float a1, float a2, float a3)
{
    GET_VAR("rtVariableSet4f");
    const float v_[4] = { a0, a1, a2, a3 };
    return varSet(_cs, "rtVariableSet4f", v, OT_FLOAT4, v_, sizeof(v_), fmtFloats(v_, 4));
}
API RTresult rtVariableSet4fv(void* h, const float* p)
{
    GET_VAR("rtVariableSet4fv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSet4fv(%s, NULL)", N(v));
    return varSet(_cs, "rtVariableSet4fv", v, OT_FLOAT4, p, 4 * sizeof(float), fmtFloats(p, 4));
}
API RTresult rtVariableGet4f(void* h, float* o0, float* o1, float* o2, float* o3)
{
    GET_VAR("rtVariableGet4f");
    float v_[4];
    RTresult r_ = varGet(_cs, "rtVariableGet4f", v, v_, sizeof(v_));
    if (o0) *o0 = v_[0];
    if (o1) *o1 = v_[1];
    if (o2) *o2 = v_[2];
    if (o3) *o3 = v_[3];
    return r_;
}
API RTresult rtVariableGet4fv(void* h, float* p)
{
    GET_VAR("rtVariableGet4fv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGet4fv(%s, NULL)", N(v));
    return varGet(_cs, "rtVariableGet4fv", v, p, 4 * sizeof(float));
}
API RTresult rtVariableSet1i(void* h, int a0)
{
    GET_VAR("rtVariableSet1i");
    const int v_[1] = { a0 };
    return varSet(_cs, "rtVariableSet1i", v, OT_INT, v_, sizeof(v_), fmtInts(v_, 1));
}
API RTresult rtVariableSet1iv(void* h, const int* p)
{
    GET_VAR("rtVariableSet1iv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSet1iv(%s, NULL)", N(v));
    return varSet(_cs, "rtVariableSet1iv", v, OT_INT, p, 1 * sizeof(int), fmtInts(p, 1));
}
API RTresult rtVariableGet1i(void* h, int* o0)
{
    GET_VAR("rtVariableGet1i");
    int v_[1];
    RTresult r_ = varGet(_cs, "rtVariableGet1i", v, v_, sizeof(v_));
    if (o0) *o0 = v_[0];
    return r_;
}
API RTresult rtVariableGet1iv(void* h, int* p)
{
    GET_VAR("rtVariableGet1iv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGet1iv(%s, NULL)", N(v));
    return varGet(_cs, "rtVariableGet1iv", v, p, 1 * sizeof(int));
}
API RTresult rtVariableSet2i(void* h, int a0, int a1)
{
    GET_VAR("rtVariableSet2i");
    const int v_[2] = { a0, a1 };
    return varSet(_cs, "rtVariableSet2i", v, OT_INT2, v_, sizeof(v_), fmtInts(v_, 2));
}
API RTresult rtVariableSet2iv(void* h, const int* p)
{
    GET_VAR("rtVariableSet2iv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSet2iv(%s, NULL)", N(v));
    return varSet(_cs, "rtVariableSet2iv", v, OT_INT2, p, 2 * sizeof(int), fmtInts(p, 2));
}
API RTresult rtVariableGet2i(void* h, int* o0, int* o1)
{
    GET_VAR("rtVariableGet2i");
    int v_[2];
    RTresult r_ = varGet(_cs, "rtVariableGet2i", v, v_, sizeof(v_));
    if (o0) *o0 = v_[0];
    if (o1) *o1 = v_[1];
    return r_;
}
API RTresult rtVariableGet2iv(void* h, int* p)
{
    GET_VAR("rtVariableGet2iv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGet2iv(%s, NULL)", N(v));
    return varGet(_cs, "rtVariableGet2iv", v, p, 2 * sizeof(int));
}
API RTresult rtVariableSet3i(void* h, int a0, int a1, int a2)
{
    GET_VAR("rtVariableSet3i");
    const int v_[3] = { a0, a1, a2 };
    return varSet(_cs, "rtVariableSet3i", v, OT_INT3, v_, sizeof(v_), fmtInts(v_, 3));
}
API RTresult rtVariableSet3iv(void* h, const int* p)
{
    GET_VAR("rtVariableSet3iv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSet3iv(%s, NULL)", N(v));
    return varSet(_cs, "rtVariableSet3iv", v, OT_INT3, p, 3 * sizeof(int), fmtInts(p, 3));
}
API RTresult rtVariableGet3i(void* h, int* o0, int* o1, int* o2)
{
    GET_VAR("rtVariableGet3i");
    int v_[3];
    RTresult r_ = varGet(_cs, "rtVariableGet3i", v, v_, sizeof(v_));
    if (o0) *o0 = v_[0];
    if (o1) *o1 = v_[1];
    if (o2) *o2 = v_[2];
    return r_;
}
API RTresult rtVariableGet3iv(void* h, int* p)
{
    GET_VAR("rtVariableGet3iv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGet3iv(%s, NULL)", N(v));
    return varGet(_cs, "rtVariableGet3iv", v, p, 3 * sizeof(int));
}
API RTresult rtVariableSet4i(void* h, int a0, int a1, int a2, int a3)
{
    GET_VAR("rtVariableSet4i");
    const int v_[4] = { a0, a1, a2, a3 };
    return varSet(_cs, "rtVariableSet4i", v, OT_INT4, v_, sizeof(v_), fmtInts(v_, 4));
}
API RTresult rtVariableSet4iv(void* h, const int* p)
{
    GET_VAR("rtVariableSet4iv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSet4iv(%s, NULL)", N(v));
    return varSet(_cs, "rtVariableSet4iv", v, OT_INT4, p, 4 * sizeof(int), fmtInts(p, 4));
}
API RTresult rtVariableGet4i(void* h, int* o0, int* o1, int* o2, int* o3)
{
    GET_VAR("rtVariableGet4i");
    int v_[4];
    RTresult r_ = varGet(_cs, "rtVariableGet4i", v, v_, sizeof(v_));
    if (o0) *o0 = v_[0];
    if (o1) *o1 = v_[1];
    if (o2) *o2 = v_[2];
    if (o3) *o3 = v_[3];
    return r_;
}
API RTresult rtVariableGet4iv(void* h, int* p)
{
    GET_VAR("rtVariableGet4iv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGet4iv(%s, NULL)", N(v));
    return varGet(_cs, "rtVariableGet4iv", v, p, 4 * sizeof(int));
}
API RTresult rtVariableSet1ui(void* h, unsigned a0)
{
    GET_VAR("rtVariableSet1ui");
    const unsigned v_[1] = { a0 };
    return varSet(_cs, "rtVariableSet1ui", v, OT_UINT, v_, sizeof(v_), fmtUints(v_, 1));
}
API RTresult rtVariableSet1uiv(void* h, const unsigned* p)
{
    GET_VAR("rtVariableSet1uiv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSet1uiv(%s, NULL)", N(v));
    return varSet(_cs, "rtVariableSet1uiv", v, OT_UINT, p, 1 * sizeof(unsigned), fmtUints(p, 1));
}
API RTresult rtVariableGet1ui(void* h, unsigned* o0)
{
    GET_VAR("rtVariableGet1ui");
    unsigned v_[1];
    RTresult r_ = varGet(_cs, "rtVariableGet1ui", v, v_, sizeof(v_));
    if (o0) *o0 = v_[0];
    return r_;
}
API RTresult rtVariableGet1uiv(void* h, unsigned* p)
{
    GET_VAR("rtVariableGet1uiv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGet1uiv(%s, NULL)", N(v));
    return varGet(_cs, "rtVariableGet1uiv", v, p, 1 * sizeof(unsigned));
}
API RTresult rtVariableSet2ui(void* h, unsigned a0, unsigned a1)
{
    GET_VAR("rtVariableSet2ui");
    const unsigned v_[2] = { a0, a1 };
    return varSet(_cs, "rtVariableSet2ui", v, OT_UINT2, v_, sizeof(v_), fmtUints(v_, 2));
}
API RTresult rtVariableSet2uiv(void* h, const unsigned* p)
{
    GET_VAR("rtVariableSet2uiv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSet2uiv(%s, NULL)", N(v));
    return varSet(_cs, "rtVariableSet2uiv", v, OT_UINT2, p, 2 * sizeof(unsigned), fmtUints(p, 2));
}
API RTresult rtVariableGet2ui(void* h, unsigned* o0, unsigned* o1)
{
    GET_VAR("rtVariableGet2ui");
    unsigned v_[2];
    RTresult r_ = varGet(_cs, "rtVariableGet2ui", v, v_, sizeof(v_));
    if (o0) *o0 = v_[0];
    if (o1) *o1 = v_[1];
    return r_;
}
API RTresult rtVariableGet2uiv(void* h, unsigned* p)
{
    GET_VAR("rtVariableGet2uiv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGet2uiv(%s, NULL)", N(v));
    return varGet(_cs, "rtVariableGet2uiv", v, p, 2 * sizeof(unsigned));
}
API RTresult rtVariableSet3ui(void* h, unsigned a0, unsigned a1, unsigned a2)
{
    GET_VAR("rtVariableSet3ui");
    const unsigned v_[3] = { a0, a1, a2 };
    return varSet(_cs, "rtVariableSet3ui", v, OT_UINT3, v_, sizeof(v_), fmtUints(v_, 3));
}
API RTresult rtVariableSet3uiv(void* h, const unsigned* p)
{
    GET_VAR("rtVariableSet3uiv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSet3uiv(%s, NULL)", N(v));
    return varSet(_cs, "rtVariableSet3uiv", v, OT_UINT3, p, 3 * sizeof(unsigned), fmtUints(p, 3));
}
API RTresult rtVariableGet3ui(void* h, unsigned* o0, unsigned* o1, unsigned* o2)
{
    GET_VAR("rtVariableGet3ui");
    unsigned v_[3];
    RTresult r_ = varGet(_cs, "rtVariableGet3ui", v, v_, sizeof(v_));
    if (o0) *o0 = v_[0];
    if (o1) *o1 = v_[1];
    if (o2) *o2 = v_[2];
    return r_;
}
API RTresult rtVariableGet3uiv(void* h, unsigned* p)
{
    GET_VAR("rtVariableGet3uiv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGet3uiv(%s, NULL)", N(v));
    return varGet(_cs, "rtVariableGet3uiv", v, p, 3 * sizeof(unsigned));
}
API RTresult rtVariableSet4ui(void* h, unsigned a0, unsigned a1, unsigned a2, unsigned a3)
{
    GET_VAR("rtVariableSet4ui");
    const unsigned v_[4] = { a0, a1, a2, a3 };
    return varSet(_cs, "rtVariableSet4ui", v, OT_UINT4, v_, sizeof(v_), fmtUints(v_, 4));
}
API RTresult rtVariableSet4uiv(void* h, const unsigned* p)
{
    GET_VAR("rtVariableSet4uiv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSet4uiv(%s, NULL)", N(v));
    return varSet(_cs, "rtVariableSet4uiv", v, OT_UINT4, p, 4 * sizeof(unsigned), fmtUints(p, 4));
}
API RTresult rtVariableGet4ui(void* h, unsigned* o0, unsigned* o1, unsigned* o2, unsigned* o3)
{
    GET_VAR("rtVariableGet4ui");
    unsigned v_[4];
    RTresult r_ = varGet(_cs, "rtVariableGet4ui", v, v_, sizeof(v_));
    if (o0) *o0 = v_[0];
    if (o1) *o1 = v_[1];
    if (o2) *o2 = v_[2];
    if (o3) *o3 = v_[3];
    return r_;
}
API RTresult rtVariableGet4uiv(void* h, unsigned* p)
{
    GET_VAR("rtVariableGet4uiv");
    if (!p) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGet4uiv(%s, NULL)", N(v));
    return varGet(_cs, "rtVariableGet4uiv", v, p, 4 * sizeof(unsigned));
}
API RTresult rtVariableSetMatrix2x2fv(void* h, int transpose, const float* m)
{
    GET_VAR("rtVariableSetMatrix2x2fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSetMatrix2x2fv(%s, NULL)", N(v));
    float s[4];   // stored row-major (2 rows x 2 columns)
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) s[i * 2 + j] = transpose ? m[j * 2 + i] : m[i * 2 + j];
    return varSet(_cs, "rtVariableSetMatrix2x2fv", v, OT_MATRIX2x2, s, sizeof(s), fmtFloats(s, 4));
}
API RTresult rtVariableGetMatrix2x2fv(void* h, int transpose, float* m)
{
    GET_VAR("rtVariableGetMatrix2x2fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetMatrix2x2fv(%s, NULL)", N(v));
    float s[4];
    RTresult r_ = varGet(_cs, "rtVariableGetMatrix2x2fv", v, s, sizeof(s));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) (transpose ? m[j * 2 + i] : m[i * 2 + j]) = s[i * 2 + j];
    return r_;
}
API RTresult rtVariableSetMatrix2x3fv(void* h, int transpose, const float* m)
{
    GET_VAR("rtVariableSetMatrix2x3fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSetMatrix2x3fv(%s, NULL)", N(v));
    float s[6];   // stored row-major (2 rows x 3 columns)
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) s[i * 3 + j] = transpose ? m[j * 2 + i] : m[i * 3 + j];
    return varSet(_cs, "rtVariableSetMatrix2x3fv", v, OT_MATRIX2x3, s, sizeof(s), fmtFloats(s, 6));
}
API RTresult rtVariableGetMatrix2x3fv(void* h, int transpose, float* m)
{
    GET_VAR("rtVariableGetMatrix2x3fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetMatrix2x3fv(%s, NULL)", N(v));
    float s[6];
    RTresult r_ = varGet(_cs, "rtVariableGetMatrix2x3fv", v, s, sizeof(s));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) (transpose ? m[j * 2 + i] : m[i * 3 + j]) = s[i * 3 + j];
    return r_;
}
API RTresult rtVariableSetMatrix2x4fv(void* h, int transpose, const float* m)
{
    GET_VAR("rtVariableSetMatrix2x4fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSetMatrix2x4fv(%s, NULL)", N(v));
    float s[8];   // stored row-major (2 rows x 4 columns)
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j) s[i * 4 + j] = transpose ? m[j * 2 + i] : m[i * 4 + j];
    return varSet(_cs, "rtVariableSetMatrix2x4fv", v, OT_MATRIX2x4, s, sizeof(s), fmtFloats(s, 8));
}
API RTresult rtVariableGetMatrix2x4fv(void* h, int transpose, float* m)
{
    GET_VAR("rtVariableGetMatrix2x4fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetMatrix2x4fv(%s, NULL)", N(v));
    float s[8];
    RTresult r_ = varGet(_cs, "rtVariableGetMatrix2x4fv", v, s, sizeof(s));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j) (transpose ? m[j * 2 + i] : m[i * 4 + j]) = s[i * 4 + j];
    return r_;
}
API RTresult rtVariableSetMatrix3x2fv(void* h, int transpose, const float* m)
{
    GET_VAR("rtVariableSetMatrix3x2fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSetMatrix3x2fv(%s, NULL)", N(v));
    float s[6];   // stored row-major (3 rows x 2 columns)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j) s[i * 2 + j] = transpose ? m[j * 3 + i] : m[i * 2 + j];
    return varSet(_cs, "rtVariableSetMatrix3x2fv", v, OT_MATRIX3x2, s, sizeof(s), fmtFloats(s, 6));
}
API RTresult rtVariableGetMatrix3x2fv(void* h, int transpose, float* m)
{
    GET_VAR("rtVariableGetMatrix3x2fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetMatrix3x2fv(%s, NULL)", N(v));
    float s[6];
    RTresult r_ = varGet(_cs, "rtVariableGetMatrix3x2fv", v, s, sizeof(s));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j) (transpose ? m[j * 3 + i] : m[i * 2 + j]) = s[i * 2 + j];
    return r_;
}
API RTresult rtVariableSetMatrix3x3fv(void* h, int transpose, const float* m)
{
    GET_VAR("rtVariableSetMatrix3x3fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSetMatrix3x3fv(%s, NULL)", N(v));
    float s[9];   // stored row-major (3 rows x 3 columns)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) s[i * 3 + j] = transpose ? m[j * 3 + i] : m[i * 3 + j];
    return varSet(_cs, "rtVariableSetMatrix3x3fv", v, OT_MATRIX3x3, s, sizeof(s), fmtFloats(s, 9));
}
API RTresult rtVariableGetMatrix3x3fv(void* h, int transpose, float* m)
{
    GET_VAR("rtVariableGetMatrix3x3fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetMatrix3x3fv(%s, NULL)", N(v));
    float s[9];
    RTresult r_ = varGet(_cs, "rtVariableGetMatrix3x3fv", v, s, sizeof(s));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) (transpose ? m[j * 3 + i] : m[i * 3 + j]) = s[i * 3 + j];
    return r_;
}
API RTresult rtVariableSetMatrix3x4fv(void* h, int transpose, const float* m)
{
    GET_VAR("rtVariableSetMatrix3x4fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSetMatrix3x4fv(%s, NULL)", N(v));
    float s[12];   // stored row-major (3 rows x 4 columns)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j) s[i * 4 + j] = transpose ? m[j * 3 + i] : m[i * 4 + j];
    return varSet(_cs, "rtVariableSetMatrix3x4fv", v, OT_MATRIX3x4, s, sizeof(s), fmtFloats(s, 12));
}
API RTresult rtVariableGetMatrix3x4fv(void* h, int transpose, float* m)
{
    GET_VAR("rtVariableGetMatrix3x4fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetMatrix3x4fv(%s, NULL)", N(v));
    float s[12];
    RTresult r_ = varGet(_cs, "rtVariableGetMatrix3x4fv", v, s, sizeof(s));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j) (transpose ? m[j * 3 + i] : m[i * 4 + j]) = s[i * 4 + j];
    return r_;
}
API RTresult rtVariableSetMatrix4x2fv(void* h, int transpose, const float* m)
{
    GET_VAR("rtVariableSetMatrix4x2fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSetMatrix4x2fv(%s, NULL)", N(v));
    float s[8];   // stored row-major (4 rows x 2 columns)
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) s[i * 2 + j] = transpose ? m[j * 4 + i] : m[i * 2 + j];
    return varSet(_cs, "rtVariableSetMatrix4x2fv", v, OT_MATRIX4x2, s, sizeof(s), fmtFloats(s, 8));
}
API RTresult rtVariableGetMatrix4x2fv(void* h, int transpose, float* m)
{
    GET_VAR("rtVariableGetMatrix4x2fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetMatrix4x2fv(%s, NULL)", N(v));
    float s[8];
    RTresult r_ = varGet(_cs, "rtVariableGetMatrix4x2fv", v, s, sizeof(s));
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) (transpose ? m[j * 4 + i] : m[i * 2 + j]) = s[i * 2 + j];
    return r_;
}
API RTresult rtVariableSetMatrix4x3fv(void* h, int transpose, const float* m)
{
    GET_VAR("rtVariableSetMatrix4x3fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSetMatrix4x3fv(%s, NULL)", N(v));
    float s[12];   // stored row-major (4 rows x 3 columns)
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j) s[i * 3 + j] = transpose ? m[j * 4 + i] : m[i * 3 + j];
    return varSet(_cs, "rtVariableSetMatrix4x3fv", v, OT_MATRIX4x3, s, sizeof(s), fmtFloats(s, 12));
}
API RTresult rtVariableGetMatrix4x3fv(void* h, int transpose, float* m)
{
    GET_VAR("rtVariableGetMatrix4x3fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetMatrix4x3fv(%s, NULL)", N(v));
    float s[12];
    RTresult r_ = varGet(_cs, "rtVariableGetMatrix4x3fv", v, s, sizeof(s));
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j) (transpose ? m[j * 4 + i] : m[i * 3 + j]) = s[i * 3 + j];
    return r_;
}
API RTresult rtVariableSetMatrix4x4fv(void* h, int transpose, const float* m)
{
    GET_VAR("rtVariableSetMatrix4x4fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableSetMatrix4x4fv(%s, NULL)", N(v));
    float s[16];   // stored row-major (4 rows x 4 columns)
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) s[i * 4 + j] = transpose ? m[j * 4 + i] : m[i * 4 + j];
    return varSet(_cs, "rtVariableSetMatrix4x4fv", v, OT_MATRIX4x4, s, sizeof(s), fmtFloats(s, 16));
}
API RTresult rtVariableGetMatrix4x4fv(void* h, int transpose, float* m)
{
    GET_VAR("rtVariableGetMatrix4x4fv");
    if (!m) return R(_cs, RT_ERROR_INVALID_VALUE, "rtVariableGetMatrix4x4fv(%s, NULL)", N(v));
    float s[16];
    RTresult r_ = varGet(_cs, "rtVariableGetMatrix4x4fv", v, s, sizeof(s));
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) (transpose ? m[j * 4 + i] : m[i * 4 + j]) = s[i * 4 + j];
    return r_;
}

// ===========================================================================
// API: Direct3D and CUDA interop - not supported (Design Garage uses OpenGL)
// ===========================================================================
API RTresult rtBufferCreateForCUDA(void* a, ...)
{
    ENTER("rtBufferCreateForCUDA");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferCreateForCUDA(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferCreateFromD3D10Resource(void* a, ...)
{
    ENTER("rtBufferCreateFromD3D10Resource");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferCreateFromD3D10Resource(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferCreateFromD3D11Resource(void* a, ...)
{
    ENTER("rtBufferCreateFromD3D11Resource");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferCreateFromD3D11Resource(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferCreateFromD3D9Resource(void* a, ...)
{
    ENTER("rtBufferCreateFromD3D9Resource");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferCreateFromD3D9Resource(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferD3D10Register(void* a, ...)
{
    ENTER("rtBufferD3D10Register");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferD3D10Register(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferD3D10Unregister(void* a, ...)
{
    ENTER("rtBufferD3D10Unregister");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferD3D10Unregister(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferD3D11Register(void* a, ...)
{
    ENTER("rtBufferD3D11Register");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferD3D11Register(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferD3D11Unregister(void* a, ...)
{
    ENTER("rtBufferD3D11Unregister");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferD3D11Unregister(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferD3D9Register(void* a, ...)
{
    ENTER("rtBufferD3D9Register");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferD3D9Register(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferD3D9Unregister(void* a, ...)
{
    ENTER("rtBufferD3D9Unregister");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferD3D9Unregister(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferGetD3D10Resource(void* a, ...)
{
    ENTER("rtBufferGetD3D10Resource");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferGetD3D10Resource(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferGetD3D11Resource(void* a, ...)
{
    ENTER("rtBufferGetD3D11Resource");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferGetD3D11Resource(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferGetD3D9Resource(void* a, ...)
{
    ENTER("rtBufferGetD3D9Resource");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferGetD3D9Resource(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferGetDevicePointer(void* a, ...)
{
    ENTER("rtBufferGetDevicePointer");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferGetDevicePointer(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferMarkDirty(void* a, ...)
{
    ENTER("rtBufferMarkDirty");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferMarkDirty(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtBufferSetDevicePointer(void* a, ...)
{
    ENTER("rtBufferSetDevicePointer");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtBufferSetDevicePointer(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtContextSetD3D10Device(void* a, ...)
{
    ENTER("rtContextSetD3D10Device");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtContextSetD3D10Device(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtContextSetD3D11Device(void* a, ...)
{
    ENTER("rtContextSetD3D11Device");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtContextSetD3D11Device(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtContextSetD3D9Device(void* a, ...)
{
    ENTER("rtContextSetD3D9Device");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtContextSetD3D9Device(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtDeviceGetD3D10Device(void* a, ...)
{
    ENTER("rtDeviceGetD3D10Device");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtDeviceGetD3D10Device(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtDeviceGetD3D11Device(void* a, ...)
{
    ENTER("rtDeviceGetD3D11Device");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtDeviceGetD3D11Device(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtDeviceGetD3D9Device(void* a, ...)
{
    ENTER("rtDeviceGetD3D9Device");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtDeviceGetD3D9Device(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtDeviceGetWGLDevice(void* a, ...)
{
    ENTER("rtDeviceGetWGLDevice");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtDeviceGetWGLDevice(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtTextureSamplerCreateFromD3D10Resource(void* a, ...)
{
    ENTER("rtTextureSamplerCreateFromD3D10Resource");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtTextureSamplerCreateFromD3D10Resource(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtTextureSamplerCreateFromD3D11Resource(void* a, ...)
{
    ENTER("rtTextureSamplerCreateFromD3D11Resource");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtTextureSamplerCreateFromD3D11Resource(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtTextureSamplerCreateFromD3D9Resource(void* a, ...)
{
    ENTER("rtTextureSamplerCreateFromD3D9Resource");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtTextureSamplerCreateFromD3D9Resource(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtTextureSamplerD3D10Register(void* a, ...)
{
    ENTER("rtTextureSamplerD3D10Register");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtTextureSamplerD3D10Register(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtTextureSamplerD3D10Unregister(void* a, ...)
{
    ENTER("rtTextureSamplerD3D10Unregister");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtTextureSamplerD3D10Unregister(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtTextureSamplerD3D11Register(void* a, ...)
{
    ENTER("rtTextureSamplerD3D11Register");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtTextureSamplerD3D11Register(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtTextureSamplerD3D11Unregister(void* a, ...)
{
    ENTER("rtTextureSamplerD3D11Unregister");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtTextureSamplerD3D11Unregister(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtTextureSamplerD3D9Register(void* a, ...)
{
    ENTER("rtTextureSamplerD3D9Register");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtTextureSamplerD3D9Register(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtTextureSamplerD3D9Unregister(void* a, ...)
{
    ENTER("rtTextureSamplerD3D9Unregister");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtTextureSamplerD3D9Unregister(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtTextureSamplerGetD3D10Resource(void* a, ...)
{
    ENTER("rtTextureSamplerGetD3D10Resource");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtTextureSamplerGetD3D10Resource(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtTextureSamplerGetD3D11Resource(void* a, ...)
{
    ENTER("rtTextureSamplerGetD3D11Resource");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtTextureSamplerGetD3D11Resource(%p, ...) [not supported by DG Bridge]", a);
}
API RTresult rtTextureSamplerGetD3D9Resource(void* a, ...)
{
    ENTER("rtTextureSamplerGetD3D9Resource");
    return R(_cs, RT_ERROR_NOT_SUPPORTED, "rtTextureSamplerGetD3D9Resource(%p, ...) [not supported by DG Bridge]", a);
}

// ===========================================================================
// Exported helper so the optixu stub can write into the same log
// ===========================================================================

API void dgbridgeLogExternal(const char* line)
{
    std::lock_guard<std::recursive_mutex> lk(g_mtx);
    ensureInit();
    logLine("%s", line ? line : "");
}

// ===========================================================================
// DLL entry
// ===========================================================================

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID reserved)
{
    if (reason == DLL_PROCESS_ATTACH) g_module = inst;
    if (reason == DLL_PROCESS_DETACH && g_log) {
        // Only touch our own state here; the process may be shutting down.
        std::fprintf(g_log, "\n==== call counts ====\n");
        for (auto& e : g_counts) std::fprintf(g_log, "%10llu  %s\n", e.second, e.first);
        std::fprintf(g_log, "==== process detaching (%s) ====\n", reserved ? "process exit" : "FreeLibrary");
        std::fclose(g_log);
        g_log = nullptr;
    }
    return TRUE;
}
