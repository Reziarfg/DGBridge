# DGBridge
A compatibility implementation for NVIDIA's free Design Garage demo from 2010

I am NOT affiliated with NVIDIA in any way. This was an academic experimental project generated through an iterative process by Anthropic's Claude Opus 5.5 LLM.
I will not provide any NVIDIA files, IP, or copyrighted material for this demo though the demo itself is freely available to this day.


> **Not affiliated with NVIDIA.** This is an independent, unofficial compatibility project. NVIDIA, OptiX and Design Garage are trademarks of NVIDIA Corporation. This repository contains no files from Design Garage or from NVIDIA's SDKs. You need your own copy of the demo and must build DG Bridge yourself.

The demo is a 32-bit program built on OptiX 3.0, and current GPUs and drivers no longer support it. DG Bridge replaces the demo's OptiX runtime (`optix.1.dll` / `optixu.1.dll`) with a compatibility layer that sends the scene to a separate 64-bit renderer using OptiX 9 and the GPU's RT cores. The finished image is handed back to the original application.

This guide explains how to build, install and test it with your own copy of Design Garage.

> [!WARNING]
>**I have not fully tested all components (e.g.: uninstall script) and I have only tested on my one machine. Execute anything in this repository at your sole risk. This worked for me on my hardware but I can make no guarantees.**

---

## 1. Requirements

| | |
|---|---|
| **OS** | Windows 10 or 11, 64-bit |
| **GPU** | NVIDIA RTX (20 series or newer). Developed and tested on an RTX 5070 Ti; other RTX cards should work but are untested. |
| **Driver** | A current NVIDIA driver that supports OptiX 9 (R590 or newer for OptiX SDK 9.1) |
| **Design Garage** | Installed from your own copy. Default location: `C:\Program Files (x86)\NVIDIA Corporation\Design Garage` |
| **Visual Studio 2022** | Community edition is fine, with the **Desktop development with C++** workload |
| **CUDA Toolkit** | 12.x or 13.x (provides `nvcc` and sets `CUDA_PATH`) |
| **OptiX SDK** | 9.1 (or 9.0) from NVIDIA's developer site, installed to its default location |

If Design Garage currently shows **"Unsupported Configuration"** on startup, you're in the right place. On RTX 50 series cards that message comes from the missing 32-bit CUDA support. Older cards may show a different message.

---

## 2. What's in the repository

| File | Purpose |
|---|---|
| `optix_stub.cpp`, `optix_stub.def` | 32-bit replacement for `optix.1.dll`; implements the OptiX 3.0 API the demo uses |
| `optixu_stub.cpp`, `optixu_stub.def` | 32-bit replacement for `optixu.1.dll` |
| `dgbridge_helper.cpp` | 64-bit render helper (OptiX 9) |
| `dgbridge_device.cu` | GPU programs: path tracer, sky, materials |
| `dgbridge_protocol.h`, `dgbridge_shared.h` | Shared data structures |
| `build_stub.bat` | Builds the two 32-bit DLLs |
| `build_helper.bat` | Builds `dgbridge_helper.exe` and `dgbridge_device.ptx` |
| `install.bat` / `uninstall.bat` | Install into, or restore, your Design Garage folder |
| `launch_custom.bat`, `launch_physical.bat`, `launch_denoise.bat` | Launchers with different settings (see section 6) |

---

## 3. Build

Every component must come from the **same version** of this repository. The stub and helper check a protocol version and refuse to work together if they don't match.

### 3a. The 32-bit stub DLLs

1. From the Start menu, open **"x86 Native Tools Command Prompt for VS 2022"**. It must be **x86**, because the demo is 32-bit.
2. Go to the repository folder and build:
   ```
   cd path\to\dgbridge
   build_stub.bat
   ```
3. You should now have `optix.1.dll` and `optixu.1.dll` in the folder.

### 3b. The 64-bit render helper

1. Open **"x64 Native Tools Command Prompt for VS 2022"**. It must be **x64**.
2. Build:
   ```
   cd path\to\dgbridge
   build_helper.bat
   ```
3. You should now have `dgbridge_helper.exe` and `dgbridge_device.ptx`.

If your OptiX SDK isn't in `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0` (or `9.0.0`), point the script to it first:
```
set "OPTIX_DIR=D:\path\to\OptiX SDK 9.1.0"
build_helper.bat
```

---

## 4. Install

1. Close Design Garage if it's running.
2. Right-click `install.bat` and choose **Run as administrator**. It needs admin rights to write into `Program Files (x86)`.
3. If Design Garage isn't in the default location, the script asks for the folder. You can also pass it directly from an administrator prompt:
   ```
   install.bat "D:\Program Files (x86)\NVIDIA Corporation\Design Garage"
   ```

The installer:
- backs up the original `optix.1.dll` and `optixu.1.dll` as `optix.1.dll.orig` and `optixu.1.dll.orig` (only the first time);
- copies in the two stub DLLs, `dgbridge_helper.exe` and `dgbridge_device.ptx`;
- adds the launcher `.bat` files next to `DesignGarage.exe`.

To undo everything, run `uninstall.bat` as administrator. It restores the original DLLs and removes the DG Bridge files.

---

## 5. Run

Start Design Garage normally, or use one of the launchers in its folder (see section 6).

**What to expect:**
- The first time you select a car or environment there's a short pause while the helper starts and uploads the scene to the GPU.
- The image starts grainy and cleans up within a second or two while the camera is still. This is progressive path tracing, the same way the original demo worked. Moving the camera or changing the paint, lighting or camera settings restarts the refinement.
- The demo's own menus and controls *mostly* work: vehicle, paint and environment selection, the lighting sliders (sun position, overcast, turbidity) and the camera settings, including depth of field. The "HQ" toggle does not currently work to disable the path tracing though this may be a trivial fix.
- Resizing the window changes the render resolution.

If you see a **moving colored gradient over a checkerboard** instead of a scene, the stub is working but the render helper isn't. See section 8.

---

## 6. Settings

Settings are environment variables read when the demo starts. The simplest way to change them is to edit `launch_custom.bat` (in the Design Garage folder after installing) and start the demo with it.

| Variable | Default | Effect |
|---|---|---|
| `DGBRIDGE_SPP` | `4` | Light paths per pixel per frame (1–64). Higher values clean up faster when the camera is still, at a lower frame rate while moving. The original demo used 1. |
| `DGBRIDGE_PHYSICAL` | `0` | `0` reproduces the original demo's light balance. `1` uses a physically correct balance (proper bounce lighting, full-size sun). |
| `DGBRIDGE_DENOISE` | `0` | `1` turns on the OptiX AI denoiser |
| `DGBRIDGE_HELPER` | `1` | `0` disables the render helper (test pattern only, for troubleshooting) |
| `DGBRIDGE_HELPER_CONSOLE` | `0` | `1` shows the helper's console window |
| `DGBRIDGE_TESTPATTERN` | `1` | Test pattern when the helper is unavailable |
| `DGBRIDGE_DUMP_BUFFERS` | `0` | `1` saves every scene buffer to disk (large; for debugging only) |
| `DGBRIDGE_DIR` | `%LOCALAPPDATA%\DGBridge` | Where logs are written |
| `DGBRIDGE_MAX_LOG` | `2000` | Maximum logged calls per OptiX function |

Ready-made launchers:
- `launch_physical.bat` uses the physically correct light balance.
- `launch_denoise.bat` turns the denoiser on.

---

## 7. Logs

Each run creates a new folder: `%LOCALAPPDATA%\DGBridge\run_YYYYMMDD_HHMMSS\`. It contains:

- `log.txt` from the 32-bit stub: every OptiX call the demo made, plus `bridge:` lines about the helper. Look for `bridge: helper ready: <your GPU>`.
- `helper_log.txt` from the 64-bit helper. It shows the active settings, the GPU, scene uploads (`scene: ... instances`) and frame progress (`frame N: WxH, P paths per pixel accumulated`).
- `scenes\` with text snapshots of the scene graph.
- `ptx\` with the shader programs the demo loaded.

---

## 8. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Still shows **"Unsupported Configuration"** | The stub DLLs aren't installed. Check that `optix.1.dll.orig` exists in the Design Garage folder, then re-run `install.bat` as administrator. |
| **Moving gradient / checkerboard** instead of the scene | The helper didn't start or failed. Check `helper_log.txt`. If it's missing, check `log.txt` for `bridge:` lines (for example, `dgbridge_helper.exe not found`). |
| `helper_log.txt` shows **protocol mismatch** | The stub and helper are from different versions. Rebuild both from the same version and reinstall. |
| `helper_log.txt` shows **cannot open ...dgbridge_device.ptx** | Re-run `build_helper.bat`, then `install.bat`. |
| `build_stub.bat` says the prompt targets x64 | Use the **x86** Native Tools prompt. |
| `build_helper.bat` says the prompt targets x86 | Use the **x64** Native Tools prompt. |
| `build_helper.bat` can't find the OptiX SDK | Set `OPTIX_DIR` as shown in section 3b. |
| `nvcc not found` / `CUDA_PATH is not set` | Install the CUDA Toolkit, then open a **new** prompt. |
| `install.bat`: "Copying failed" | Close Design Garage and try again. |
| Errors mentioning OptiX initialization | Update your NVIDIA driver; the OptiX version you built against needs a matching driver. |

To get the original behavior back at any time, run `uninstall.bat` as administrator.

---

## Known limitations

- Every frame is copied between the 64-bit helper and the 32-bit demo, which limits frame rate at large window sizes. Raising `DGBRIDGE_SPP` makes better use of the GPU per frame.
- Only tested on an RTX 5070 Ti so far.
- HQ toggle not working but doesn't seem to matter much with modern GPUs
