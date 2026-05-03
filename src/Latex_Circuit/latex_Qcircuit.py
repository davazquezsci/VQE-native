from pathlib import Path
import subprocess
import shutil

def circuito_a_latex_imagen(
    qc,
    nombre_salida="circuito",
    carpeta_salida=".",
    compilar_pdf=True,
    convertir_png=True,
    dpi=300,
    limpiar_aux=True
):
    carpeta_salida = Path(carpeta_salida).resolve()
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    tex_path = carpeta_salida / f"{nombre_salida}.tex"
    pdf_path = carpeta_salida / f"{nombre_salida}.pdf"
    png_path = carpeta_salida / f"{nombre_salida}.png"
    log_path = carpeta_salida / f"{nombre_salida}.log"
    aux_path = carpeta_salida / f"{nombre_salida}.aux"

    # 🔥 IMPORTANTE: esto ya es un documento completo
    try:
        latex_circuito = qc.draw(output="latex_source")
    except Exception as e:
        raise RuntimeError(
            "Qiskit no pudo generar 'latex_source'. "
            "Verifica que esté instalado pylatexenc."
        ) from e

    # ✅ NO envolver, guardar tal cual
    tex_path.write_text(latex_circuito, encoding="utf-8")

    rutas = {
        "tex": str(tex_path),
        "pdf": None,
        "png": None,
        "log": str(log_path),
    }

    # Compilar PDF
    if compilar_pdf:
        if shutil.which("pdflatex") is None:
            raise RuntimeError("No se encontró 'pdflatex'.")

        resultado = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name
            ],
            cwd=str(tex_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if resultado.returncode != 0:
            raise RuntimeError(
                "Falló pdflatex\n\n"
                f"LOG: {log_path}\n\n"
                f"{resultado.stdout}"
            )

        rutas["pdf"] = str(pdf_path)

    # Convertir a PNG
    if compilar_pdf and convertir_png:
        if shutil.which("pdftocairo") is None:
            raise RuntimeError("No se encontró 'pdftocairo'.")

        subprocess.run(
            [
                "pdftocairo",
                "-png",
                "-singlefile",
                "-r", str(dpi),
                str(pdf_path),
                str(carpeta_salida / nombre_salida)
            ],
            cwd=str(tex_path.parent),
            check=True
        )

        rutas["png"] = str(png_path)

    # Limpiar
    if limpiar_aux:
        for archivo in [aux_path, log_path]:
            if archivo.exists():
                archivo.unlink()

    return rutas