@echo off
rem Starts Design Garage with your own DG Bridge settings. Edit the values below.

rem Light paths per pixel per frame (1-64). Higher = image cleans up faster
rem when still, lower frame rate while moving. Default 4; the original demo used 1.
set DGBRIDGE_SPP=4

rem 1 = physically correct light balance, 0 = as in the original demo (default)
set DGBRIDGE_PHYSICAL=0

rem 1 = OptiX AI denoiser on, 0 = off (default)
set DGBRIDGE_DENOISE=0

cd /d "%~dp0"
start "" DesignGarage.exe
