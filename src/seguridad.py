"""
Sistema de seguridad del proyecto: código de 4 dígitos en Morse con tarjetas.

Cada tarjeta es blanca con 5 símbolos negros en fila: punto = círculo, raya = rectángulo
alargado. La tarjeta se coloca dentro del recuadro guía del centro de la imagen.

Detección (Temas 3 y 5, Sesión 3):
    escala de grises -> suavizado gaussiano -> umbral fijo (invertido) -> apertura
    morfológica -> componentes conexas con área mínima -> rectángulo mínimo de cada una
    -> punto/raya por relación de aspecto r = w/h -> orden de izquierda a derecha por c_x

Decodificador:
    un dígito se acepta si se lee igual N fotogramas seguidos; después se espera a que la
    tarjeta salga del recuadro; con 4 dígitos se compara S con S*.

Uso:
    python seguridad.py
Teclas: q / ESC = salir, b = borrar la secuencia, s = guardar los pasos de la imagen.
"""

import os
import time
import cv2
import numpy as np

# ---------------------------------------------------------------- parámetros
CAMERA_INDEX = 0
WIDTH, HEIGHT = 1280, 720
SECRET = (1, 9, 0, 5)          # contraseña S* (cambiadla)

GUIDE_W, GUIDE_H = 560, 260    # recuadro guía (px) donde se pone la tarjeta
BLUR_SIGMA = 1.5               # sigma del filtro gaussiano
THRESH = 100                   # umbral fijo: píxel < THRESH -> símbolo (ajustar con la luz de la sala)
OPEN_K = 5                     # tamaño del elemento estructurante de la apertura
MIN_AREA = 150                 # área mínima (px) de una componente para ser símbolo
DOT_MAX_R = 1.4                # r = w/h <= DOT_MAX_R -> punto
DASH_MIN_R = 1.8               # r = w/h >= DASH_MIN_R -> raya (entre ambos: lectura dudosa)

N_STABLE = 15                  # fotogramas seguidos con la misma lectura para aceptar un dígito
N_EMPTY = 10                   # fotogramas sin símbolos para considerar que la tarjeta ha salido
T_RESET = 20.0                 # segundos sin actividad que borran la secuencia a medias
T_WRONG_WAIT = 4.0             # segundos de espera tras un código erróneo antes de leer otra tarjeta

MORSE = {
    ".----": 1, "..---": 2, "...--": 3, "....-": 4, ".....": 5,
    "-....": 6, "--...": 7, "---..": 8, "----.": 9, "-----": 0,
}


# ---------------------------------------------------------------- detector de símbolos
def guide_box(frame):
    """Recuadro centrado donde el usuario debe colocar la tarjeta."""
    H_img, W_img = frame.shape[:2]
    x = (W_img - GUIDE_W) // 2
    y = (H_img - GUIDE_H) // 2
    return x, y, GUIDE_W, GUIDE_H


def detect_symbols(roi_bgr):
    """
    Devuelve (lectura, simbolos, pasos):
      lectura  -> cadena de '.' y '-' si hay una fila válida de 5 símbolos, si no None
      simbolos -> lista de (x, y, w, h, tipo) de las componentes aceptadas
      pasos    -> imágenes intermedias (para el informe)
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), BLUR_SIGMA)
    # Umbral fijo invertido: los símbolos negros quedan en blanco (255)
    binary = np.where(blur < THRESH, 255, 0).astype(np.uint8)
    # Apertura: elimina puntos sueltos sin deformar los símbolos
    kernel = np.ones((OPEN_K, OPEN_K), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Componentes conexas con un área mínima y su rectángulo mínimo
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    comps = []
    for i in range(1, n):                      # la 0 es el fondo
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        # Se descartan componentes que tocan el borde del recuadro (p.ej. dedos, borde de la tarjeta)
        if x == 0 or y == 0 or x + w >= opened.shape[1] or y + h >= opened.shape[0]:
            continue
        comps.append((x, y, w, h))

    symbols = []
    reading = None
    if comps:
        # Los símbolos de una tarjeta tienen la misma altura y están en la misma fila.
        # La fila se ajusta con una recta por mínimos cuadrados (c_y = a c_x + b), así se
        # admite que la tarjeta esté algo inclinada.
        h_med = np.median([c[3] for c in comps])
        row = [c for c in comps if 0.5 * h_med <= c[3] <= 1.5 * h_med]
        if len(row) >= 2:
            cxs = np.array([c[0] + c[2] / 2 for c in row])
            cys = np.array([c[1] + c[3] / 2 for c in row])
            a, b = np.polyfit(cxs, cys, 1)
            row = [c for c, cx, cy in zip(row, cxs, cys) if abs(cy - (a * cx + b)) <= 0.5 * h_med]
        row.sort(key=lambda c: c[0] + c[2] / 2)      # izquierda -> derecha por c_x

        chars = []
        for (x, y, w, h) in row:
            r = w / h
            if r <= DOT_MAX_R:
                t = "."
            elif r >= DASH_MIN_R:
                t = "-"
            else:
                t = "?"                               # dudoso
            symbols.append((x, y, w, h, t))
            chars.append(t)
        code = "".join(chars)
        if len(code) == 5 and "?" not in code:
            reading = code

    # Imagen de componentes coloreadas (para el informe)
    comp_vis = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h, t) in symbols:
        color = (0, 200, 0) if t == "." else (0, 0, 255) if t == "-" else (0, 200, 255)
        cv2.rectangle(comp_vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(comp_vis, t, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    steps = {"1_gris": gray, "2_gaussiano": blur, "3_umbral": binary,
             "4_apertura": opened, "5_componentes": comp_vis}
    return reading, symbols, steps


# ---------------------------------------------------------------- decodificador
class MorseDecoder:
    """Memoriza hasta 4 dígitos y concede o deniega el acceso."""

    def __init__(self, secret=SECRET):
        self.secret = tuple(secret)
        self.blocked_until = 0.0    # tras un código erróneo no se lee nada hasta este instante
        self.reset()

    def reset(self):
        self.sequence = []          # S = (d1, d2, d3, d4)
        self.candidate = None
        self.count = 0
        self.waiting_removal = False
        self.empty_count = 0
        self.last_activity = time.time()

    def update(self, reading, n_symbols):
        """
        reading: cadena Morse de 5 símbolos o None. n_symbols: símbolos vistos en el recuadro.
        Devuelve un evento: None, 'digito', 'correcto', 'erroneo', 'caducado'.
        """
        # Tras un código erróneo: espera de T_WRONG_WAIT s (y que se retire la tarjeta)
        if time.time() < self.blocked_until:
            return None

        # Secuencia a medias sin actividad durante T_RESET -> se olvida
        if self.sequence and time.time() - self.last_activity > T_RESET:
            self.reset()
            return "caducado"

        # Tras aceptar un dígito, esperar a que la tarjeta salga (evita contarla dos veces)
        if self.waiting_removal:
            self.empty_count = self.empty_count + 1 if n_symbols == 0 else 0
            if self.empty_count >= N_EMPTY:
                self.waiting_removal = False
            return None

        digit = MORSE.get(reading) if reading else None
        if digit is None:
            self.candidate, self.count = None, 0
            return None

        # Misma lectura N_STABLE fotogramas seguidos -> dígito aceptado
        if digit == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = digit, 1
        if self.count < N_STABLE:
            return None

        self.sequence.append(digit)
        self.candidate, self.count = None, 0
        self.waiting_removal, self.empty_count = True, 0
        self.last_activity = time.time()

        if len(self.sequence) < 4:
            return "digito"
        ok = tuple(self.sequence) == self.secret     # acceso <=> S = S*
        self.reset()
        if not ok:
            self.blocked_until = time.time() + T_WRONG_WAIT
            self.waiting_removal = True               # además hay que retirar la última tarjeta
        return "correcto" if ok else "erroneo"

    def wait_left(self):
        """Segundos que quedan de espera tras un código erróneo."""
        return max(0.0, self.blocked_until - time.time())

    def progress(self):
        return len(self.sequence), (self.count / N_STABLE if self.candidate is not None else 0.0)


# ---------------------------------------------------------------- bucle del bloqueo
def run_security(cap, map1=None, map2=None, on_event=None):
    """
    Bucle del bloque de seguridad. Devuelve True si se introduce el código correcto,
    False si se sale con q/ESC. on_event(evento) permite enlazar los avisos por voz.
    """
    decoder = MorseDecoder()
    message, message_until = "", 0.0
    t_prev, fps = time.time(), 0.0
    os.makedirs("pasos_seguridad", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            return False
        if map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        gx, gy, gw, gh = guide_box(frame)
        roi = frame[gy:gy + gh, gx:gx + gw]
        reading, symbols, steps = detect_symbols(roi)
        event = decoder.update(reading, len(symbols))

        if event == "digito":
            message = f"Digito {len(decoder.sequence)} leido. Retire la tarjeta."
        elif event == "erroneo":
            message = "CODIGO ERRONEO"
        elif event == "caducado":
            message = "Secuencia borrada por inactividad"
        elif event == "correcto":
            message = "ACCESO CONCEDIDO"
        if event:
            message_until = time.time() + (T_WRONG_WAIT if event == "erroneo" else 2.0)
            if on_event:
                on_event(event)

        # ---- dibujo
        view = frame.copy()
        cv2.rectangle(view, (gx, gy), (gx + gw, gy + gh), (255, 255, 255), 2)
        cv2.putText(view, "Coloque la tarjeta aqui", (gx, gy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for (x, y, w, h, t) in symbols:
            color = (0, 200, 0) if t == "." else (0, 0, 255) if t == "-" else (0, 200, 255)
            cv2.rectangle(view, (gx + x, gy + y), (gx + x + w, gy + y + h), color, 2)
        n_done, frac = decoder.progress()
        stars = " ".join(["*"] * n_done + ["_"] * (4 - n_done))
        cv2.putText(view, f"Codigo: {stars}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        if frac > 0:   # barra de progreso de la lectura estable
            cv2.rectangle(view, (gx, gy + gh + 10), (gx + int(gw * min(frac, 1)), gy + gh + 22), (0, 255, 0), -1)
        if decoder.wait_left() > 0:
            cv2.putText(view, f"Espere {decoder.wait_left():.0f} s para volver a intentarlo",
                        (gx, gy + gh + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif decoder.waiting_removal:
            cv2.putText(view, "Retire la tarjeta", (gx, gy + gh + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if time.time() < message_until:
            color = (0, 0, 255) if message == "CODIGO ERRONEO" else (0, 255, 0)
            cv2.putText(view, message, (20, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        cv2.putText(view, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Bloqueo", view)

        if event == "correcto":
            cv2.waitKey(1000)
            cv2.destroyWindow("Bloqueo")
            return True

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            cv2.destroyWindow("Bloqueo")
            return False
        if key == ord('b'):
            decoder.reset()
            message, message_until = "Secuencia borrada", time.time() + 2.0
        if key == ord('s'):
            stamp = time.strftime("%H%M%S")
            cv2.imwrite(f"pasos_seguridad/{stamp}_0_original.png", roi)
            for name, img in steps.items():
                cv2.imwrite(f"pasos_seguridad/{stamp}_{name}.png", img)
            message, message_until = "Pasos guardados en pasos_seguridad/", time.time() + 2.0


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        print(f"No se puede abrir la cámara (índice {CAMERA_INDEX}).")
        return
    try:
        from tracker import load_undistort_maps     # misma corrección que el resto del proyecto
        map1, map2 = load_undistort_maps()
    except (ImportError, FileNotFoundError):
        print("Aviso: sin calibracion.npz, se trabaja sin corregir la distorsión.")
        map1 = map2 = None

    ok = run_security(cap, map1, map2, on_event=lambda e: print("Evento:", e))
    print("Acceso concedido" if ok else "Salida sin acceso")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
