# Onyco-Scan
OnycoScan is a multi-modal clinical decision support system for subungual Acral Lentiginous Melanoma triage. It combines class-weighted DenseNet121 imaging, trauma-evolution rule logic, and guideline-grounded RAG to reduce missed melanoma and unnecessary biopsies caused by the “Trauma Trap.”

# OnycoScan

### Neuro-Symbolic Clinical Decision Support for Nail Melanoma Triage

---

## Executive Summary

OnycoScan is a multi-modal clinical decision support system (CDSS) designed to improve triage accuracy for subungual Acral Lentiginous Melanoma (ALM) while reducing unnecessary biopsies of benign nail conditions.

Subungual melanoma is rare but highly lethal and frequently misdiagnosed due to its visual similarity to subungual hematoma. Published literature documents diagnostic delays of up to nine months, often driven by a cognitive bias known as the “Trauma Trap” — where pigmented nail lesions are prematurely attributed to minor injury.

OnycoScan addresses this high-risk diagnostic gap by combining deep learning, rule-based reasoning, and guideline-grounded retrieval into a unified, explainable AI system.

This project demonstrates advanced modeling, neuro-symbolic architecture design, and clinical AI alignment principles.

---

## Business & Clinical Problem

Healthcare systems face two competing risks:

1. **False negatives** → Missed melanoma leading to late-stage diagnosis
2. **False positives** → Unnecessary nail matrix biopsies causing pain, dystrophy, and cost

Most AI dermatology systems treat melanoma detection as a purely visual classification task. However, nail melanoma diagnosis often depends on:

* Trauma history
* Lesion evolution over time
* Guideline-defined red flags
* Clinical context beyond the image

A purely image-based model cannot resolve this ambiguity reliably.

OnycoScan reframes the task as a **triage optimization problem**, not just image classification.

---

## System Architecture

OnycoScan uses a four-layer modular architecture:

### 1 > Clinical Risk Re-Bucketization

Original six-class nail dataset restructured into three clinically meaningful categories:

* Melanoma (Target)
* Healthy
* Other Pathology (confounders)

This prevents binary collapse and better reflects real-world triage complexity.

---

### 2 >  Class-Weighted DenseNet121 Visual Engine

* Transfer learning with ImageNet initialization
* Two-phase training (frozen backbone → fine-tuning top layers)
* Class-weighted categorical cross-entropy to prioritize melanoma recall
* Data augmentation to mitigate small-sample overfitting

**Objective:** Maximize sensitivity for melanoma under class imbalance constraints.

Pilot dataset: 91 dermoscopic nail images
Internal validation accuracy: 100% (acknowledged as proof-of-concept, not deployment-ready)

---

### 3 >  Trauma–Evolution Heuristic Layer (Neuro-Symbolic Bridge)

To address the documented “Trauma Trap,” OnycoScan integrates structured rule logic:

Inputs:

* CNN probability distribution
* Trauma history
* Temporal lesion evolution

Example logic:
If high melanoma probability + recent trauma + no documented widening → recommend short-interval follow-up instead of immediate biopsy.

This hybrid approach reduces reliance on raw probabilities and mirrors real clinical reasoning patterns.

---

### 4 >  Guideline-Grounded Retrieval-Augmented Generation (RAG)

To improve transparency and clinician trust:

* Authoritative dermatology guidelines segmented into passages
* Embedded using SentenceTransformers
* Indexed via FAISS
* Top-k guideline passages retrieved at inference
* LLM generates structured, guideline-aligned explanation

Instead of outputting:

> Class 0: 0.92

The system generates:

> Irregular widening longitudinal melanonychia with periungual extension. Current guidelines recommend prompt biopsy.

This transforms a black-box classifier into an interpretable decision-support tool.

---

## Technical Stack

* Python
* TensorFlow / Keras
* DenseNet121 (transfer learning)
* Scikit-learn (class weighting)
* SentenceTransformers
* FAISS
* Pandas / NumPy

---

## Machine Learning Design Decisions

* Class weighting to prioritize recall in high-stakes screening context
* Three-class reformulation to prevent pathological label simplification
* Neuro-symbolic integration to mitigate known diagnostic failure modes
* Retrieval-constrained generation to reduce hallucination risk

This project reflects deliberate modeling tradeoffs aligned with clinical risk management principles.

---

## Limitations

* Small single-source dataset (n=91)
* No external validation cohort
* No prospective clinical trial
* Heuristic rules not outcome-calibrated

This system is a research prototype, not a medical device.

---

## Future Roadmap

* External multi-center validation
* ROC-AUC and per-class sensitivity benchmarking
* Calibration curve analysis
* Grad-CAM interpretability
* Prospective workflow simulation
* Deployment as clinician-facing triage interface

---

### What This Project Demonstrates

Through OnycoScan, I demonstrate:

* Advanced transfer learning implementation
* Handling of class imbalance in medical datasets
* Neuro-symbolic AI system design
* Retrieval-augmented generation architecture
* Alignment of AI modeling decisions with real-world clinical risk

This project reflects a systems-level approach to healthcare AI.  moving beyond prediction toward decision-support design.

