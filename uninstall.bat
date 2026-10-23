@echo off
rem Restores the original OptiX DLLs and removes the DG Bridge helper files.
rem Right-click > Run as administrator, or pass the folder as an argument.

setlocal
net session >nul 2>&1 || goto need_admin
set "DG=%~1"
if "%DG%"=="" set "DG=%ProgramFiles(x86)%\NVIDIA Corporation\Design Garage"
if exist "%DG%\optix.1.dll.orig" goto found
echo No backup found at:
echo   %DG%
set /p "DG=Enter the Design Garage folder: "
if not exist "%DG%\optix.1.dll.orig" goto no_backup

:found
copy /y "%DG%\optix.1.dll.orig" "%DG%\optix.1.dll" >nul || goto copy_failed
copy /y "%DG%\optixu.1.dll.orig" "%DG%\optixu.1.dll" >nul || goto copy_failed
del /q "%DG%\dgbridge_helper.exe" "%DG%\dgbridge_device.ptx" "%DG%\launch_physical.bat" "%DG%\launch_no_denoise.bat" "%DG%\launch_denoise.bat" "%DG%\launch_custom.bat" >nul 2>nul
echo Original OptiX DLLs restored, helper files removed.
pause
exit /b 0

:need_admin
echo Please right-click uninstall.bat and choose "Run as administrator".
pause
exit /b 1
:no_backup
echo No optix.1.dll.orig backup found in that folder.
pause
exit /b 1
:copy_failed
echo Copying failed. Is Design Garage still running?
pause
exit /b 1
