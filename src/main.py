"""
Monitor de postura y distancia — programa principal.

Fases:
  1. BLOQUEO     -> seguridad.run_security (código Morse con 4 tarjetas)
  2. SELECCIÓN   -> pegatinas E, C, H (con tracker), ROI fijo del ordenador y pegatina de escala
  3. VIGILANCIA  -> tracker + ángulos + distancia + avisos por voz + información en el vídeo

Teclas en vigilancia: q / ESC = salir, r = volver a seleccionar, l = bloquear (pide el código).
Al salir se muestra el informe de la sesión (porcentaje de tiempo con buena postura y
gráfica de los ángulos).

Archivos necesarios en la misma carpeta: calibracion.npz, tracker.py, seguridad.py,
medidas.py, avisos.py.
"""

import os
import time
import unicodedata
import cv2
import numpy as np

from tracker import (load_undistort_maps, create_trackers, select_regions, draw_pc_roi,
                     CAMERA_INDEX, WIDTH, HEIGHT)
from seguridad import run_security
from medidas import posture_angles, detect_screen_edge, screen_x_at, measure_scale, distance_cm
from avisos import Voice, AlertManager

T_PAUSE = 45 * 60          # recordatorio de pausa (s)
GREEN, RED, WHITE, YELLOW = (0, 200, 0), (0, 0, 255), (255, 255, 255), (0, 255, 255)


def read_frame(cap, map1, map2):
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


def capture_photo(cap, map1, map2):
    """
    Vídeo en directo hasta que se pulsa ESPACIO: así la persona se coloca con calma y la
    foto para seleccionar las pegatinas se hace cuando está lista.
    """
    while True:
        frame = read_frame(cap, map1, map2)
        if frame is None:
            return None
        view = frame.copy()
        cv2.rectangle(view, (10, 10), (1100, 60), (0, 0, 0), -1)
        cv2.putText(view, "Colocate y pulsa ESPACIO para hacer la foto (q = salir)", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, YELLOW, 2)
        cv2.imshow("Foto para seleccionar", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            cv2.destroyWindow("Foto para seleccionar")
            return frame
        if key == ord('q') or key == 27:
            cv2.destroyWindow("Foto para seleccionar")
            return None


def setup(cap, map1, map2):
    """Fase de selección. Devuelve trackers, ROI del ordenador, borde de pantalla y escala."""
    frame = capture_photo(cap, map1, map2)
    if frame is None:
        return None                                       # se ha salido con q / ESC
    trackers, pc_roi = create_trackers(frame)
    ref_roi = select_regions(frame, ["Pegatina de escala"])[0]

    scale, ref_mask = measure_scale(frame, ref_roi)
    person_x = trackers[0].center[0]                      # posición de E
    screen, edges = detect_screen_edge(frame, pc_roi, person_x)
    if scale is None:
        print("Aviso: no se encuentra la pegatina de escala; no se medirá la distancia.")
    if screen is None:
        print("Aviso: no se encuentra el borde de la pantalla; no se medirá la distancia.")

    # Imágenes intermedias para el informe (secuencia de transformación)
    os.makedirs("pasos_medidas", exist_ok=True)
    x, y, w, h = pc_roi
    cv2.imwrite("pasos_medidas/1_roi_ordenador.png", frame[y:y + h, x:x + w])
    cv2.imwrite("pasos_medidas/2_canny.png", edges)
    cv2.imwrite("pasos_medidas/3_mascara_escala.png", ref_mask)
    return trackers, pc_roi, screen, scale


def ascii_text(text):
    """cv2.putText no dibuja tildes: se quitan solo para mostrarlas en pantalla."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def draw_overlay(frame, trackers, pc_roi, screen, alerts, theta_cv, theta_t, d, fps, eta):
    """Información superpuesta en el vídeo."""
    for t in trackers:
        t.draw(frame)
    draw_pc_roi(frame, pc_roi)

    E, C, H = [tuple(int(v) for v in t.center) for t in trackers]
    posture_color = RED if alerts.bad_posture() else GREEN
    cv2.line(frame, E, C, RED if alerts.rules["cuello"].bad else GREEN, 3)
    cv2.line(frame, C, H, RED if alerts.rules["espalda"].bad else GREEN, 3)

    if screen is not None:
        x1, y1, x2, y2 = screen
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), YELLOW, 2)
        if d is not None:
            xs = int(screen_x_at(screen, E[1]))
            dist_color = RED if alerts.bad_distance() else GREEN
            cv2.arrowedLine(frame, E, (xs, E[1]), dist_color, 2, tipLength=0.03)
            cv2.arrowedLine(frame, (xs, E[1]), E, dist_color, 2, tipLength=0.03)

    # Panel de valores
    lines = [f"FPS: {fps:.1f}",
             f"theta_CV: {theta_cv:.1f} deg" if theta_cv is not None else "theta_CV: --",
             f"theta_T:  {theta_t:.1f} deg" if theta_t is not None else "theta_T: --",
             f"d: {d:.1f} cm" if d is not None else "d: --",
             f"Buena postura: {eta:.0f} %" if eta is not None else "Buena postura: --"]
    cv2.rectangle(frame, (10, 10), (330, 30 + 32 * len(lines)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        color = posture_color if i in (1, 2) else WHITE
        cv2.putText(frame, text, (20, 40 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Último aviso, visible 3 s
    if time.time() - alerts.last_message_t < 3.0:
        cv2.putText(frame, ascii_text(alerts.last_message).upper(), (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, RED, 4)


def session_report(alerts):
    eta = alerts.good_ratio()
    print("\n===== Informe de la sesión =====")
    print(f"Tiempo vigilado: {alerts.t_total / 60:.1f} min")
    print(f"Tiempo con buena postura: {eta:.1f} %" if eta is not None else "Sin medidas.")
    try:
        import matplotlib.pyplot as plt
        h = [r for r in alerts.history if r[1] is not None]
        if not h:
            return
        t0 = h[0][0]
        t = [(r[0] - t0) / 60 for r in h]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t, [r[1] for r in h], label="theta_CV (cuello)")
        ax.plot(t, [r[2] for r in h], label="theta_T (tronco)")
        ax.set_xlabel("Tiempo (min)")
        ax.set_ylabel("Ángulo (grados)")
        ax.set_title(f"Sesión: {eta:.0f} % del tiempo con buena postura")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig("informe_sesion.png", dpi=150)
        print("Gráfica guardada en informe_sesion.png")
        plt.show()
    except ImportError:
        pass


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        print(f"No se puede abrir la cámara (índice {CAMERA_INDEX}).")
        return
    map1, map2 = load_undistort_maps()
    voice = Voice()

    def on_security_event(event):
        if event == "erroneo":
            voice.say("Código erróneo")
        elif event == "correcto":
            voice.say("Acceso concedido")

    # 1. BLOQUEO
    if not run_security(cap, map1, map2, on_event=on_security_event):
        cap.release()
        return

    # 2. SELECCIÓN
    selection = setup(cap, map1, map2)
    if selection is None:
        cap.release()
        cv2.destroyAllWindows()
        return
    trackers, pc_roi, screen, scale = selection
    alerts = AlertManager(voice)
    session_start = last_pause = time.time()

    # 3. VIGILANCIA
    t_prev, fps = time.time(), 0.0
    while True:
        frame = read_frame(cap, map1, map2)
        if frame is None:
            break
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        found = [t.update(frame_hsv)[1] for t in trackers]
        E, C, H = [t.center for t in trackers]
        if all(found):
            theta_cv, theta_t = posture_angles(E, C, H)
        else:
            theta_cv = theta_t = None                    # alguna pegatina perdida: no se mide
        d = distance_cm(E, screen, scale) if found[0] else None
        alerts.update(theta_cv, theta_t, d)

        # Recordatorio de pausas (ampliación)
        now = time.time()
        if now - last_pause >= T_PAUSE:
            voice.say("Llevas mucho tiempo sentado. Haz una pausa")
            alerts.last_message, alerts.last_message_t = "Haz una pausa", now
            last_pause = now

        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        draw_overlay(frame, trackers, pc_roi, screen, alerts, theta_cv, theta_t, d, fps,
                     alerts.good_ratio())
        cv2.imshow("Monitor de postura", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('r'):
            selection = setup(cap, map1, map2)
            if selection is None:
                break
            trackers, pc_roi, screen, scale = selection
        if key == ord('l'):                              # bloqueo rápido (ampliación)
            cv2.destroyWindow("Monitor de postura")
            if not run_security(cap, map1, map2, on_event=on_security_event):
                break
            selection = setup(cap, map1, map2)
            if selection is None:
                break
            trackers, pc_roi, screen, scale = selection

    cap.release()
    cv2.destroyAllWindows()
    session_report(alerts)


if __name__ == "__main__":
    main()
