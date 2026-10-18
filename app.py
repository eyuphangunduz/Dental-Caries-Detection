"""
app.py

Simple web interface for the demo: upload a panoramic X-ray in the browser and
see the carious teeth marked with their tooth number and severity.
Uses the same pipeline as predict.py.

Usage:  python app.py   -> then open the link printed in the console
"""
import gradio as gr

from predict import analyze, load_models, summary_text

print("Loading models...")
LOADED = load_models()


def run(image):
    if image is None:
        return None, "Lütfen bir panoramik röntgen yükleyin."
    annotated, findings, n_teeth = analyze(image, LOADED)
    return annotated, summary_text(findings, n_teeth)


demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="Panoramik diş röntgeni"),
    outputs=[
        gr.Image(type="pil", label="Sonuç"),
        gr.Textbox(label="Bulgular", lines=6),
    ],
    title="Panoramik Röntgende Diş Çürüğü Tespiti",
    description=(
        "Bir panoramik diş röntgeni yükleyin. Sistem dişleri bulur, numaralandırır ve her dişi "
        "sağlıklı / çürük / derin çürük olarak sınıflandırır. "
        "Turuncu kutu: çürük, kırmızı kutu: derin çürük. "
        "Bu bir araştırma projesidir; tıbbi teşhis yerine geçmez."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()