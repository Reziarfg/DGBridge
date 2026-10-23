@echo off
rem Starts Design Garage with the DG Bridge AI denoiser turned on.
set DGBRIDGE_DENOISE=1
cd /d "%~dp0"
start "" DesignGarage.exe
