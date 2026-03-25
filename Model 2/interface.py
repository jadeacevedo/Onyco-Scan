# @title 🏥 The OnycoScan Clinical Agent (Interactive)
import importlib.util
import io
import math
import os
import pathlib

import numpy as np
import tensorflow as tf
from PIL import Image

try:
    from IPython.display import display, clear_output
    import ipywidgets as widgets
except ImportError:
    widgets = None
    display = None
    clear_output = None


def _load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASEDIR = pathlib.Path(__file__).resolve().parent
CONFIG = None
try:
    CONFIG = _load_module("model2_config", BASEDIR / "1_configuration.py")
except Exception:
    pass

MODEL_PATH = BASEDIR / "onycoscan_weighted.keras"


def load_model(model_path=None):
    path = pathlib.Path(model_path or MODEL_PATH)
    if not path.exists():
        alt_path = path.with_suffix('.h5')
        if alt_path.exists():
            path = alt_path

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    try:
        return tf.keras.models.load_model(str(path))
    except Exception as e:
        # If .keras fails, fallback to h5 if available
        if path.suffix != '.h5':
            alt = path.with_suffix('.h5')
            if alt.exists():
                return tf.keras.models.load_model(str(alt))
        raise


CLASSES = ["Acral_Melanoma", "Healthy_Nail", "Other_Condition"]


def generate_diagnosis(history, visual_class, visual_conf):
    is_trauma = "Yes" in history.get("trauma", "No")
    is_changing = "Yes" in history.get("evolution", "No Change")
    is_family_risk = "Yes" in history.get("family_hx", "No")

    if visual_class == "Acral_Melanoma":
        if is_trauma and not is_changing:
            diagnosis = "Suspicious for Subungual Hematoma vs. Melanoma."
            action = "Perform dermoscopy. If pigment moves with nail growth, it is hematoma. Else biopsy."
            risk_level = "High (Rule out malignancy)"
        elif is_trauma and is_changing:
            diagnosis = "High probability of Acral Lentiginous Melanoma."
            action = "Immediate dermatology referral for matrix biopsy."
            risk_level = "Critical"
        else:
            diagnosis = "Diagnosis: Subungual Melanoma likely."
            action = f"Visual confidence {visual_conf:.1f}%. Urgent biopsy."
            risk_level = "Critical"
    elif visual_class == "Healthy_Nail":
        if is_changing or "Pain" in history.get("pain", "None"):
            diagnosis = "Visually benign but symptomatic."
            action = "Monitor for 4 weeks and reassess."
            risk_level = "Low-Moderate"
        else:
            diagnosis = "Nail appears healthy."
            action = "Routine follow-up."
            risk_level = "Low"
    else:
        diagnosis = "Dystrophic Nail / Other condition."
        action = "Evaluate for fungal infection, psoriasis, or trauma."
        risk_level = "Moderate"

    return diagnosis, action, risk_level


class PatientIntake:
    def __init__(self):
        self.history = {}
        self.questions = [
            {"key": "age", "q": "Patient Age:", "type": "text"},
            {"key": "sex", "q": "Biological Sex:", "type": "dropdown", "opts": ["Male", "Female", "Intersex"]},
            {"key": "skin_type", "q": "Skin Type (Fitzpatrick):", "type": "dropdown", "opts": ["I (Pale)", "II", "III", "IV", "V", "VI (Deeply Pigmented)"]},
            {"key": "duration", "q": "Duration of pigmentation (months):", "type": "text"},
            {"key": "evolution", "q": "Has it changed? (Widening/Darkening):", "type": "dropdown", "opts": ["No Change", "Yes - Widening", "Yes - Darkening", "Yes - Both"]},
            {"key": "trauma", "q": "History of Trauma to this nail?", "type": "dropdown", "opts": ["No", "Yes (Recent)", "Yes (Old)"]},
            {"key": "family_hx", "q": "Family history of Melanoma?", "type": "dropdown", "opts": ["No", "Yes (1st Degree)", "Yes (Extended)"]},
            {"key": "pain", "q": "Symptoms (Pain/Bleeding/Ulcer):", "type": "dropdown", "opts": ["None", "Pain", "Bleeding", "Ulceration"]},
        ]

    def render_form(self):
        if widgets is None:
            raise RuntimeError("ipywidgets is not installed, cannot render form")

        self.widgets = {}
        ui_elements = []
        for item in self.questions:
            label = widgets.Label(item["q"])
            w = widgets.Text(placeholder="Type here...") if item["type"] == "text" else widgets.Dropdown(options=item["opts"])
            self.widgets[item["key"]] = w
            ui_elements.append(widgets.VBox([label, w]))

        return widgets.VBox(ui_elements)

    def get_data(self):
        return {k: v.value for k, v in self.widgets.items()}


def is_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return 'zmq' in str(type(ip))
    except Exception:
        return False


def run_terminal(model):
    print("\n=== OnycoScan Terminal Mode ===")
    history = {}
    history["age"] = input("Patient age: ").strip()
    history["sex"] = input("Biological sex: ").strip()
    history["skin_type"] = input("Skin type: ").strip()
    history["duration"] = input("Duration in months: ").strip()
    history["evolution"] = input("Evolution (No Change/Yes - Widening/Yes - Darkening/Yes - Both): ").strip()
    history["trauma"] = input("Trauma history (No/Yes (Recent)/Yes (Old)): ").strip()
    history["family_hx"] = input("Family history of Melanoma (No/Yes): ").strip()
    history["pain"] = input("Pain/Bleeding/Ulcer (None/Pain/Bleeding/Ulceration): ").strip()

    image_path = input("Local image path (e.g. /path/to/nail.jpg): ").strip()
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img_raw = Image.open(image_path).convert("RGB")
    img_arr = tf.keras.preprocessing.image.img_to_array(img_raw.resize((224, 224)))
    img_arr = tf.keras.applications.densenet.preprocess_input(img_arr)
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr, verbose=0)
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]) * 100)
    visual = CLASSES[idx]

    diagnosis_txt, action, risk = generate_diagnosis(history, visual, conf)

    print("\n=== Prediction ===")
    print(f"Detected: {visual} ({conf:.2f}%)")
    print("=== Clinical Diagnosis ===")
    print(f"Risk Level: {risk}")
    print(f"Diagnosis: {diagnosis_txt}")
    print(f"Plan: {action}")


def run_noninteractive(model_path=None):
    model = load_model(model_path)
    print("Model loaded.")
    return model


if __name__ == "__main__":
    model = None
    try:
        model = load_model()
        print("✅ Model Loaded")
    except Exception as exc:
        print("⚠️ Model load failed:", exc)
        model = None

    if not is_notebook() or widgets is None or display is None:
        print("Running in terminal mode.")
        if model is None:
            raise RuntimeError("Model is required for terminal mode.")
        run_terminal(model)
    else:
        print("Running in notebook mode.")
        intake = PatientIntake()
        form_ui = intake.render_form()
        upload_btn = widgets.FileUpload(accept='image/*', multiple=False)
        run_btn = widgets.Button(description='🧠 Run Analysis', button_style='danger')
        out = widgets.Output()

        def on_run_click(_):
            with out:
                clear_output()
                history = intake.get_data()
                if not upload_btn.value:
                    print("❌ Please upload an image first!")
                    return

                up_obj = list(upload_btn.value.values())[0]
                img_raw = Image.open(io.BytesIO(up_obj['content'])).convert('RGB')
                img_arr = tf.keras.preprocessing.image.img_to_array(img_raw.resize((224, 224)))
                img_arr = tf.keras.applications.densenet.preprocess_input(img_arr)
                img_arr = np.expand_dims(img_arr, axis=0)

                if model is None:
                    print("⚠️ Model not loaded, cannot predict")
                    return

                preds = model.predict(img_arr, verbose=0)
                idx = int(np.argmax(preds[0]))
                conf = float(np.max(preds[0]) * 100)
                visual = CLASSES[idx]

                print(f"Detected: {visual} ({conf:.2f}%)")
                diagnosis, action, risk = generate_diagnosis(history, visual, conf)
                print("\n=== AI Clinical Synthesis ===")
                print(f"Risk Level: {risk}")
                print(f"Diagnosis: {diagnosis}")
                print(f"Plan: {action}")

        run_btn.on_click(on_run_click)

        display(form_ui)
        display(widgets.Label("📸 Upload Nail Photo:"))
        display(upload_btn)
        display(run_btn)
        display(out)
