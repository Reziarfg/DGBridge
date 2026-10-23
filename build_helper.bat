@echo off
rem Builds the 64-bit render helper (dgbridge_helper.exe) and its GPU programs.
rem Run from the "x64 Native Tools Command Prompt for VS 2022".
rem Finds OptiX SDK 9.1 / 9.0 in the default location, or set OPTIX_DIR first.

setlocal
cd /d "%~dp0"

set "SDKROOT=C:\ProgramData\NVIDIA Corporation"
if "%OPTIX_DIR%"=="" if exist "%SDKROOT%\OptiX SDK 9.1.0\include\optix.h" set "OPTIX_DIR=%SDKROOT%\OptiX SDK 9.1.0"
if "%OPTIX_DIR%"=="" if exist "%SDKROOT%\OptiX SDK 9.0.0\include\optix.h" set "OPTIX_DIR=%SDKROOT%\OptiX SDK 9.0.0"
if "%OPTIX_DIR%"=="" goto no_optix
if not exist "%OPTIX_DIR%\include\optix.h" goto no_optix
where cl >nul 2>nul || goto no_cl
if /i not "%VSCMD_ARG_TGT_ARCH%"=="x64" goto wrong_arch
where nvcc >nul 2>nul || goto no_nvcc
if "%CUDA_PATH%"=="" goto no_cuda_path

echo Using OptiX SDK: %OPTIX_DIR%
echo.
echo [1/2] Compiling GPU programs...
nvcc -ptx -m64 -arch=compute_75 --use_fast_math -I"%OPTIX_DIR%\include" -I. dgbridge_device.cu -o dgbridge_device.ptx
if errorlevel 1 goto failed

echo [2/2] Compiling helper...
cl /nologo /EHsc /O2 /MD /std:c++17 /W3 /I"%OPTIX_DIR%\include" /I"%CUDA_PATH%\include" /I. dgbridge_helper.cpp /Fe:dgbridge_helper.exe /link /LIBPATH:"%CUDA_PATH%\lib\x64" cudart_static.lib advapi32.lib cfgmgr32.lib
if errorlevel 1 goto failed
del /q *.obj >nul 2>nul

echo.
echo Build OK. Now run install.bat as administrator.
exit /b 0

:no_optix
echo Could not find the OptiX SDK 9.1 or 9.0. Set OPTIX_DIR and run again.
exit /b 1
:no_cl
echo cl.exe not found. Use the "x64 Native Tools Command Prompt for VS 2022".
exit /b 1
:wrong_arch
echo This prompt targets "%VSCMD_ARG_TGT_ARCH%". Use the "x64 Native Tools Command Prompt for VS 2022".
exit /b 1
:no_nvcc
echo nvcc not found. Install the CUDA Toolkit, then open a new prompt.
exit /b 1
:no_cuda_path
echo CUDA_PATH is not set. Reinstall the CUDA Toolkit or set CUDA_PATH.
exit /b 1
:failed
echo.
echo Build failed - please send me the full output above.
exit /b 1
