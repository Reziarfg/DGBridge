@echo off
rem Starts Design Garage with DG Bridge's physically correct light balance
rem (proper diffuse bounces and full-size sun) instead of the original demo's.
set DGBRIDGE_PHYSICAL=1
cd /d "%~dp0"
start "" DesignGarage.exe
