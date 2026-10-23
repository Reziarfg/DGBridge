@echo off
rem OPTIONAL: rebuilds the 32-bit stub DLLs from source. Prebuilt copies are
rem in the "prebuilt" folder, so you only need this if you change the code.
rem Run from the "x86 Native Tools Command Prompt for VS 2022" (NOT x64).

setlocal
cd /d "%~dp0"
where cl >nul 2>nul || goto no_cl
if /i not "%VSCMD_ARG_TGT_ARCH%"=="x86" goto wrong_arch

echo [1/2] Building optix.1.dll...
cl /nologo /O2 /MT /EHsc /std:c++17 /W3 /I. /LD optix_stub.cpp /Fe:optix.1.dll /link /DEF:optix_stub.def
if errorlevel 1 goto failed
echo [2/2] Building optixu.1.dll...
cl /nologo /O2 /MT /EHsc /std:c++17 /W3 /LD optixu_stub.cpp /Fe:optixu.1.dll /link /DEF:optixu_stub.def
if errorlevel 1 goto failed
del /q *.obj *.exp *.lib >nul 2>nul
echo.
echo Build OK. install.bat will use these instead of the prebuilt copies.
exit /b 0

:no_cl
echo cl.exe not found. Use the "x86 Native Tools Command Prompt for VS 2022".
exit /b 1
:wrong_arch
echo This prompt targets "%VSCMD_ARG_TGT_ARCH%". Use the "x86 Native Tools Command Prompt for VS 2022".
exit /b 1
:failed
echo.
echo Build failed - please send me the full output above.
exit /b 1
