@echo off
cd /d "%~dp0"
call env_PM\Scripts\activate.bat
set PATH=%PATH%;E:\AI_Projects\PacMan\swigwin
cmd /k