@echo off
setlocal
cd /d "%~dp0"
title BBR Vector Studio v1.01
set "STUDIO_DIR=%~dp0studio"
set "PYTHONPATH=%STUDIO_DIR%;%PYTHONPATH%"
where py >nul 2>nul
if %errorlevel%==0 (
  set PY=py -3
) else (
  set PY=python
)
%PY% -c "import PySide6, OpenGL, numpy, texture2ddecoder, lz4, PIL" >nul 2>nul
if errorlevel 1 (
  echo Installing required Python packages...
  %PY% -m pip install -r "%~dp0requirements.txt"
  if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    pause
    exit /b 1
  )
)
%PY% "%STUDIO_DIR%\bbr_vector_studio.py" %*
if errorlevel 1 pause
endlocal
