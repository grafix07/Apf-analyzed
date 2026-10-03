@echo off
setlocal
cd /d "%~dp0\.."
where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)
%PY% -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
%PY% -m pip install pyinstaller
if errorlevel 1 exit /b 1
%PY% -m PyInstaller --noconfirm --clean --windowed ^
  --name "BBR Vector Studio" ^
  --icon "assets\BBR_Vector_Studio.ico" ^
  --add-data "assets;assets" ^
  --paths "studio" ^
  "studio\bbr_vector_studio.py"
if errorlevel 1 (
  echo EXE build failed.
  pause
  exit /b 1
)
echo.
echo EXE created in dist\BBR Vector Studio\
pause
endlocal
