@echo off
setlocal

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting Administrator permission...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

@REM delete all contents at C:\Temp
del /f /q "C:\Temp\*.*"
for /d %%D in ("C:\Temp\*") do rd /s /q "%%D"

@REM recreate iceoryx2 if needed
mkdir "C:\Temp\iceoryx2" 2>nul
