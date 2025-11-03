@echo off
cd /d "%~dp0"
call env_KFM\Scripts\activate.bat
set PATH=%PATH%;C:\AI\KungFuMaster-ML\swigwin
cmd /k