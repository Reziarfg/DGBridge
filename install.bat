@echo off
rem Installs DG Bridge stage 4b into the Design Garage folder.
rem Backs up the original OptiX DLLs as *.orig (only the first time).
rem Right-click > Run as administrator, or from an admin prompt:
rem   install.bat "G:\Program Files (x86)\NVIDIA Corporation\Design Garage"

setlocal
net session >nul 2>&1 || goto need_admin
set "HERE=%~dp0"

set "DG=%~1"
if "%DG%"=="" set "DG=%ProgramFiles(x86)%\NVIDIA Corporation\Design Garage"
if exist "%DG%\DesignGarage.exe" goto found
echo Design Garage was not found at:
echo   %DG%
set /p "DG=Enter the Design Garage folder: "
if not exist "%DG%\DesignGarage.exe" goto not_found

:found
echo Installing into: %DG%

if not exist "%DG%\optix.1.dll.orig" copy /y "%DG%\optix.1.dll" "%DG%\optix.1.dll.orig" >nul
if not exist "%DG%\optixu.1.dll.orig" copy /y "%DG%\optixu.1.dll" "%DG%\optixu.1.dll.orig" >nul
if not exist "%DG%\optix.1.dll.orig" goto backup_failed

set "SRC=%HERE%"
if not exist "%SRC%optix.1.dll" set "SRC=%HERE%prebuilt\"
copy /y "%SRC%optix.1.dll" "%DG%\optix.1.dll" >nul || goto copy_failed
copy /y "%SRC%optixu.1.dll" "%DG%\optixu.1.dll" >nul || goto copy_failed
echo   stub DLLs from %SRC%

if not exist "%HERE%dgbridge_helper.exe" goto no_helper
copy /y "%HERE%dgbridge_helper.exe" "%DG%\dgbridge_helper.exe" >nul || goto copy_failed
set "PTX=%HERE%dgbridge_device.ptx"
if not exist "%PTX%" set "PTX=%HERE%prebuilt\dgbridge_device.ptx"
copy /y "%PTX%" "%DG%\dgbridge_device.ptx" >nul || goto copy_failed
copy /y "%HERE%launch_physical.bat" "%DG%\launch_physical.bat" >nul
copy /y "%HERE%launch_denoise.bat" "%DG%\launch_denoise.bat" >nul
copy /y "%HERE%launch_custom.bat" "%DG%\launch_custom.bat" >nul
del /q "%DG%\launch_no_denoise.bat" >nul 2>nul
echo   render helper and GPU programs
echo.
echo Done. Logs will appear in: %LOCALAPPDATA%\DGBridge
pause
exit /b 0

:no_helper
echo.
echo WARNING: dgbridge_helper.exe was not found - run build_helper.bat first.
echo The stub is installed, but it can only show the test pattern.
pause
exit /b 1

:need_admin
echo Please right-click install.bat and choose "Run as administrator".
pause
exit /b 1
:not_found
echo DesignGarage.exe was not found in that folder.
pause
exit /b 1
:backup_failed
echo Could not back up the original OptiX DLLs. Nothing was changed.
pause
exit /b 1
:copy_failed
echo Copying failed. Is Design Garage still running?
pause
exit /b 1
