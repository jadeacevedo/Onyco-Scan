# @title 4. Build Clinical RAG Engine (Updated)
from sentence_transformers import SentenceTransformer
import faiss
import json

# 1. Expanded Knowledge Base to cover "Other" conditions
guidelines_data = [
    # --- MELANOMA (Class 0) ---
    {
        "class_group": "Acral_Melanoma",
        "signal": "Hutchinson sign",
        "description": "Pigment extending onto the periungual skin (cuticle).",
        "risk": "Critical",
        "recommendation": "Urgent dermatology referral for biopsy.",
    },
    {
        "class_group": "Acral_Melanoma",
        "signal": "Longitudinal Melanonychia",
        "description": "Irregular pigmented band >3mm, variegated colors, or widening proximally.",
        "risk": "Critical",
        "recommendation": "Dermoscopic evaluation required.",
    },
    # --- HEALTHY (Class 1) ---
    {
        "class_group": "Healthy_Nail",
        "signal": "Clear nail plate",
        "description": "Smooth texture, uniform color, no pigmentation or dystrophy.",
        "risk": "Low",
        "recommendation": "Routine monitoring.",
    },
    # --- OTHER CONDITIONS (Class 2) ---
    {
        "class_group": "Other_Condition",
        "signal": "Nail Clubbing",
        "description": "Bulbous enlargement of fingertips with convex nail plate.",
        "risk": "Moderate",
        "recommendation": "Screen for cardiac or pulmonary causes.",
    },
    {
        "class_group": "Other_Condition",
        "signal": "Nail Pitting",
        "description": "Small depressions (ice-pick) on nail surface.",
        "risk": "Low",
        "recommendation": "Common in psoriasis or alopecia areata.",
    },
    {
        "class_group": "Other_Condition",
        "signal": "Blue Finger",
        "description": "Cyanotic discoloration.",
        "risk": "High (Acute)",
        "recommendation": "Check oxygen saturation and circulation immediately.",
    }
]

# Save for App
with open("guidelines_3way.json", "w") as f:
    json.dump(guidelines_data, f, indent=2)

# 2. Vector Indexing
print("🧠 Initializing RAG Index...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# We embed: "Class Group + Signal + Description" for better retrieval context
corpus = [f"{g['class_group']} {g['signal']}: {g['description']}" for g in guidelines_data]
embeddings = encoder.encode(corpus)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def retrieve_guidance(query_text):
    """Retrieves guideline based on query."""
    q_vec = encoder.encode([query_text])
    D, I = index.search(q_vec, k=1)
    return guidelines_data[I[0][0]]

print("✅ RAG Engine Ready with expanded 'Other' class definitions.")