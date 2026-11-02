@echo off
REM ===================================================================
REM  Analisis de algoritmos en el contexto de la bibliometria
REM  Lanzador de un clic: arranca la aplicacion y abre el navegador.
REM ===================================================================

cd /d "%~dp0"
title Analisis de algoritmos - aplicacion web

REM Usa el entorno virtual del proyecto si existe. Si no, el Python del
REM sistema. Sin esto, un doble clic correria con el Python global y no
REM encontraria Flask, que es el error mas comun al entregar el proyecto.
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

REM -m app.lanzar y no la ruta del archivo: el servidor importa desde src/,
REM y eso solo funciona si Python arranca desde la raiz del proyecto.
"%PY%" -m app.lanzar

REM Si algo fallo, la ventana se queda abierta para poder leer el error.
if errorlevel 1 (
    echo.
    echo   La aplicacion termino con un error.
    echo   Revisa el mensaje de arriba.
    echo.
    pause
)
