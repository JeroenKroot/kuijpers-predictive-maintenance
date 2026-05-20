@echo off
cd /d "%~dp0"

echo ===============================================
echo Kuijpers Predictive Maintenance App
echo ===============================================
echo.

echo Projectmap:
cd
echo.

echo Virtual environment activeren...
call "%~dp0.venv\Scripts\activate.bat"

echo.
echo Gebruikte Python:
where python
python --version
echo.

echo Controle benodigde packages...
python -c "import streamlit; import librosa; import sklearn; import pandas; import numpy; print('Alle basispackages zijn beschikbaar')"

echo.
echo Streamlit-app wordt gestart op poort 8503...
echo.

start "" /B python -m streamlit run app.py --server.port 8503 --server.headless true

echo Wachten tot Streamlit is opgestart...
timeout /t 5 /nobreak > nul

echo.
echo App openen in Chrome app-modus...
start chrome --app=http://127.0.0.1:8503

echo.
echo De app draait nu lokaal.
echo Sluit dit venster pas als je de app wilt stoppen.
echo.

pause