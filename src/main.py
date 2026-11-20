"""Monitor de postura y distancia - programa principal.

Bucle único con máquina de estados:  BLOQUEO -> SELECCION -> VIGILANCIA.
Cada fase es una función que procesa UN fotograma y devuelve el estado siguiente.

Teclas:
    q / Esc  salir
    n        (provisional) pasar a la fase siguiente
    b        bloqueo rápido: volver a BLOQUEO
"""
import os
import time

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
CAMERA_INDEX = 1            # 0 suele ser la webcam integrada
FRAME_W, FRAME_H = 1280, 720

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_PATH = os.path.join(ROOT, "data", "calibracion.npz")   # claves: K, dist
CALIB_W, CALIB_H = 1920, 1080   # resolución a la que se hicieron las fotos de calibración

GRABAR = False              # True para guardar el vídeo de la demo
FPS_GRABACION = 20.0        # pon aquí los FPS reales medidos con el esqueleto
VIDEO_PATH = os.path.join(ROOT, "output", "demo.avi")

WINDOW = "Monitor de postura"

# Estados
BLOQUEO = "BLOQUEO"
SELECCION = "SELECCION"
VIGILANCIA = "VIGILANCIA"
SIGUIENTE = {BLOQUEO: SELECCION, SELECCION: VIGILANCIA, VIGILANCIA: VIGILANCIA}

# Colores (BGR)
BLANCO, VERDE, ROJO, AMARILLO = (255, 255, 255), (0, 255, 0), (0, 0, 255), (0, 255, 255)


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def cargar_calibracion(path, size):
    """Devuelve (map1, map2) para remap, o (None, None) si aún no hay calibración."""
    if not os.path.exists(path):
        print(f"[AVISO] No se encuentra {path}: se trabaja SIN corregir la distorsión.")
        return None, None
    data = np.load(path)
    dist = data["dist"]

    # K se calibró a CALIB_W x CALIB_H; hay que escalarla a la resolución real (size).
    # fx, fy, cx, cy están en píxeles, así que se multiplican por el factor de escala.
    # Los coeficientes de distorsión no dependen de la resolución: no se tocan.
    w, h = size
    s = w / CALIB_W
    if abs(h / CALIB_H - s) > 1e-3:
        print(f"[AVISO] {w}x{h} no tiene la misma proporción que {CALIB_W}x{CALIB_H}: "
              "la calibración no es válida, se trabaja SIN corregir la distorsión.")
        return None, None
    K = data["K"].copy()
    K[0, 0] *= s; K[1, 1] *= s   # fx, fy
    K[0, 2] *= s; K[1, 2] *= s   # cx, cy

    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_16SC2)
    print(f"[OK] Calibración cargada desde {path} (escala {s:.3f})")
    return map1, map2


def texto(img, msg, pos, color=BLANCO, escala=0.8):
    """putText con borde negro para que se lea sobre cualquier fondo."""
    cv2.putText(img, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Fases (por ahora solo rotulan la pantalla; se rellenarán en las próximas sesiones)
# Cada una recibe el fotograma y su diccionario de memoria, y devuelve el estado siguiente.
# ----------------------------------------------------------------------------
def fase_bloqueo(frame, mem):
    # TODO sesión 2: detector de símbolos + decodificador de la secuencia
    texto(frame, "BLOQUEO - ensena las tarjetas", (20, 80), ROJO)
    texto(frame, f"Codigo: {''.join(mem['digitos']):_<4}", (20, 120))
    return BLOQUEO


def fase_seleccion(frame, mem):
    # TODO sesión 3: selectROI de E, C, H y ordenador + inicializar trackers/Kalman
    texto(frame, "SELECCION - marca E, C, H y el ordenador", (20, 80), AMARILLO)
    return SELECCION


def fase_vigilancia(frame, mem):
    # TODO sesiones 3-4: meanShift + Kalman, ángulos, distancia y avisos
    texto(frame, "VIGILANCIA", (20, 80), VERDE)
    return VIGILANCIA


FASES = {BLOQUEO: fase_bloqueo, SELECCION: fase_seleccion, VIGILANCIA: fase_vigilancia}


def nueva_memoria():
    """Memoria de cada fase. Vive fuera del bucle para sobrevivir entre fotogramas."""
    return {
        BLOQUEO: {"digitos": []},
        SELECCION: {},
        VIGILANCIA: {},
    }


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara (índice {CAMERA_INDEX}).")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # La cámara puede no aceptar la resolución pedida: usamos la real
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolución: {w}x{h}")

    map1, map2 = cargar_calibracion(CALIB_PATH, (w, h))

    writer = None
    if GRABAR:
        os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS_GRABACION, (w, h))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    estado = BLOQUEO
    mem = nueva_memoria()
    t_prev = time.perf_counter()
    fps = 0.0

    try:
        while True:
            # 1. Leer
            ret, frame = cap.read()
            if not ret:
                print("No se recibe imagen (¿cámara desconectada?).")
                break

            # 2. Corregir distorsión (sin flip: invertiría la lectura del Morse)
            if map1 is not None:
                frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)

            # 3. Fase actual
            estado = FASES[estado](frame, mem[estado])

            # 4. FPS (suavizados) y estado, comunes a todas las fases
            ahora = time.perf_counter()
            fps_inst = 1.0 / max(ahora - t_prev, 1e-6)
            t_prev = ahora
            fps = fps_inst if fps == 0 else 0.9 * fps + 0.1 * fps_inst
            texto(frame, f"FPS: {fps:.1f}", (20, 40), VERDE)
            texto(frame, estado, (w - 230, 40), AMARILLO)

            # 5. Mostrar y grabar (después de dibujar, para que salga en el vídeo)
            cv2.imshow(WINDOW, frame)
            if writer is not None:
                writer.write(frame)

            # 6. Teclado
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('n'):                      # provisional
                estado = SIGUIENTE[estado]
            elif key == ord('b'):                      # bloqueo rápido
                estado = BLOQUEO
                mem = nueva_memoria()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
