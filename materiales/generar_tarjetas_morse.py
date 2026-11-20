"""Genera las 6 tarjetas Morse imprescindibles para el sistema de seguridad.

Al girar una tarjeta 180° los símbolos se leen en orden inverso, de modo que cada tarjeta
sirve para dos dígitos: 1/9, 2/8, 3/7, 4/6. El 5 y el 0 son simétricos.

Cada tarjeta ocupa una hoja A4 apaisada completa (297 x 210 mm): 6 hojas en total.
Diseño: fondo blanco, símbolos negros rellenos, punto = círculo de diámetro PUNTO,
raya = rectángulo de H x 3H, separación entre símbolos = H.

Uso: python generar_tarjetas_morse.py [salida.pdf]
"""
import sys

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

MORSE = {
    "1": ".----", "2": "..---", "3": "...--", "4": "....-", "5": ".....",
    "6": "-....", "7": "--...", "8": "---..", "9": "----.", "0": "-----",
}
# (dígito al derecho, dígito al girarla 180°)
TARJETAS = [("1", "9"), ("2", "8"), ("3", "7"), ("4", "6"), ("5", "5"), ("0", "0")]

H = 13 * mm           # alto de la raya
PUNTO = 20 * mm       # diámetro del punto
RAYA = 3 * H          # largo de la raya (relación de aspecto 3:1)
HUECO = H             # separación entre símbolos

PAGE_W, PAGE_H = landscape(A4)   # 297 x 210 mm
CARD_H = PAGE_H


def dibujar_tarjeta(c, tarjeta, y0):
    """Dibuja una tarjeta ocupando la franja [y0, y0 + CARD_H] de la página."""
    digito, girado = tarjeta
    codigo = MORSE[digito]
    ancho_total = sum(PUNTO if s == "." else RAYA for s in codigo) + HUECO * (len(codigo) - 1)
    x = (PAGE_W - ancho_total) / 2
    yc = y0 + CARD_H / 2

    c.setFillColorRGB(0, 0, 0)
    for s in codigo:
        if s == ".":
            c.circle(x + PUNTO / 2, yc, PUNTO / 2, stroke=0, fill=1)
            x += PUNTO
        else:
            c.rect(x, yc - H / 2, RAYA, H, stroke=0, fill=1)
            x += RAYA
        x += HUECO

    # Número en gris muy claro en la esquina (queda por encima del umbral, se ve como blanco)
    c.setFillColorRGB(0.88, 0.88, 0.88)
    c.setFont("Helvetica", 10)
    # Abajo a la izquierda: dígito que se lee así; arriba a la derecha, boca abajo: dígito al girarla
    c.drawString(8 * mm, y0 + 6 * mm, digito)
    if girado != digito:
        c.saveState()
        c.translate(PAGE_W - 8 * mm, y0 + CARD_H - 6 * mm)
        c.rotate(180)
        c.drawString(0, 0, girado)
        c.restoreState()


def main(salida="tarjetas_morse.pdf"):
    c = canvas.Canvas(salida, pagesize=landscape(A4))
    c.setTitle("Tarjetas Morse - Monitor de postura y distancia")
    for tarjeta in TARJETAS:
        dibujar_tarjeta(c, tarjeta, 0)
        c.showPage()
    c.save()


if __name__ == "__main__":
    main(*sys.argv[1:])
