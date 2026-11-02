"""
Lanzador de la aplicación para uso local
=========================================

Arranca el servidor y abre el navegador cuando la aplicación ya está lista,
de modo que quien la ejecuta no tiene que escribir ninguna dirección.

POR QUÉ NO BASTA CON ESPERAR UNOS SEGUNDOS
------------------------------------------
Lo fácil sería esperar dos o tres segundos y abrir el navegador. Es una mala
solución: el servidor tarda lo que tarde en cargar el corpus y calcular el IDF
de los 20 abstracts, y eso depende de la máquina. Si el equipo va lento, el
navegador abre antes de tiempo y el evaluador ve un error de conexión; si va
rápido, se pierden segundos esperando de más.

Este lanzador no adivina: abre un socket contra el puerto una vez cada cuarto
de segundo y solo abre el navegador cuando el servidor acepta la conexión. Es
la diferencia entre suponer que ya está listo y comprobarlo.

USO
---
    python -m app.lanzar

o, en Windows, doble clic en «Iniciar aplicacion.bat».
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser

HOST = "127.0.0.1"
PUERTO = int(os.environ.get("PORT", "5000"))
URL = f"http://{HOST}:{PUERTO}"

ESPERA_MAXIMA = 90.0     # segundos antes de rendirse
INTERVALO = 0.25         # cada cuánto se sondea el puerto


def _puerto_responde(host: str, puerto: int, timeout: float = 0.4) -> bool:
    """True si hay algo escuchando y acepta una conexión."""
    try:
        with socket.create_connection((host, puerto), timeout=timeout):
            return True
    except OSError:
        return False


def _puerto_ocupado(puerto: int) -> bool:
    """
    True si el puerto ya está en uso antes de arrancar.

    Vale la pena comprobarlo: si quedó una instancia anterior corriendo, Flask
    falla con un error de socket que no dice nada útil a quien solo hizo doble
    clic en un icono.
    """
    return _puerto_responde(HOST, puerto, timeout=0.2)


def _abrir_cuando_este_listo() -> None:
    """Sondea el puerto y abre el navegador en cuanto el servidor responde."""
    inicio = time.monotonic()
    while time.monotonic() - inicio < ESPERA_MAXIMA:
        if _puerto_responde(HOST, PUERTO):
            transcurrido = time.monotonic() - inicio
            print(f"   La aplicación está lista ({transcurrido:.1f} s). "
                  f"Abriendo el navegador…")
            webbrowser.open(URL)
            return
        time.sleep(INTERVALO)
    print(f"   El servidor no respondió en {ESPERA_MAXIMA:.0f} s. "
          f"Abre manualmente {URL}")


def main() -> None:
    if _puerto_ocupado(PUERTO):
        print(f"\n  El puerto {PUERTO} ya está ocupado.")
        print(f"  Puede que la aplicación ya esté corriendo: abre {URL}")
        print(f"  Si no, cierra la ventana anterior y vuelve a intentarlo.\n")
        webbrowser.open(URL)
        sys.exit(1)

    print()
    print("  Análisis de algoritmos en el contexto de la bibliometría")
    print("  " + "-" * 54)
    print("  Iniciando la aplicación. El navegador se abrirá solo.")
    print(f"  Si no se abre, entra a {URL}")
    print("  Para cerrarla: Ctrl+C en esta ventana.")
    print()

    # El hilo se marca como daemon para que no impida cerrar el programa con
    # Ctrl+C mientras sigue sondeando.
    threading.Thread(target=_abrir_cuando_este_listo, daemon=True).start()

    # Se importa aquí y no arriba: importar el servidor carga el corpus y
    # calcula el IDF, y queremos que el aviso de arriba ya esté impreso cuando
    # eso ocurra, para que la ventana no se vea congelada sin explicación.
    from app.servidor import app
    app.run(host=HOST, port=PUERTO, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
