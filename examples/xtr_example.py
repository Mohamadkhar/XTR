from lightning_ir import BiEncoderModule
from lightning_ir.models.xtr import XTRConfig, XTRModel, XTRTokenizer
import torch

# Set random seed for reproducibility
torch.manual_seed(42)

model = XTRModel.from_pretrained("bert-base-uncased")
tokenizer = XTRTokenizer.from_pretrained("bert-base-uncased")
module = BiEncoderModule(model=model, evaluation_metrics=["nDCG@10"])

# Score queries and documents
queries = ["What is the capital of France?"]
documents = [
    [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.", 
        "Madrid is the capital of Spain.",
        "Rome is the capital of Italy.",
    ]
]

print("XTR Model Scoring Example")
print("=" * 40)
print(f"Query: {queries[0]}")
print("\nDocuments:")
for i, doc in enumerate(documents[0]):
    print(f"{i+1}. {doc}")

print("\nScoring with XTR (attention-weighted late interaction)...")
scores = module.score(queries=queries, docs=documents)
print(f"Scores: {scores}")

# Print scores with documents for easier interpretation
print("\nResults:")
for i, (doc, score) in enumerate(zip(documents[0], scores[0])):
    print(f"{i+1}. Score: {score.item():.4f} - {doc}")
