"""One-time fetch of the semantic-scoring embedding model into the repo.

Run this ONCE on a machine that can reach huggingface.co (your local
Windows machine), then commit the resulting models_local/ directory:

    python scripts\\fetch_embedding_model.py
    git add models_local
    git commit -m "Vendor all-MiniLM-L6-v2 for offline semantic scoring"
    git push origin main

After that, evaluator.py loads the model from models_local/ first, so
sandboxed environments that can't reach huggingface.co (like the Claude
skill container) still compute the exact same semantic score.

The saved model is roughly 90 MB, under GitHub's 100 MB per-file limit,
so plain git works and Git LFS is deliberately NOT used (LFS storage
endpoints may not be reachable from sandboxed environments).
"""

import os

from sentence_transformers import SentenceTransformer

TARGET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models_local",
    "all-MiniLM-L6-v2",
)


def main():
    if os.path.isdir(TARGET_DIR) and os.listdir(TARGET_DIR):
        print(f"Already vendored at {TARGET_DIR} - nothing to do.")
        return
    print("Downloading all-MiniLM-L6-v2 from huggingface.co ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    os.makedirs(TARGET_DIR, exist_ok=True)
    model.save(TARGET_DIR)
    print(f"Saved to {TARGET_DIR}")
    print("Now commit models_local/ and push (see the docstring above).")


if __name__ == "__main__":
    main()
