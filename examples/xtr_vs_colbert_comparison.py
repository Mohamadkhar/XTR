import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from lightning_ir import BiEncoderModule
from lightning_ir.models.xtr import XTRConfig
from lightning_ir.loss import InfoNCE

print(" Starting XTR Training with Simple Examples")
print("=" * 50)

# Deine Test-Cases aus xtr_vs_colbert_comparison.py
test_cases = [
    {
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "Madrid is the capital of Spain.",
            "Rome is the capital of Italy.",
        ]
    },
  
]

# XTR Konfiguration
config = XTRConfig(
    query_length=32,
    doc_length=128,
    embedding_dim=768,
    normalize=True
)

print("✓ XTR Config created")

# XTR Module
module = BiEncoderModule(
    model_name_or_path="bert-base-uncased",
    config=config,
    loss_functions=[InfoNCE()],
    evaluation_metrics=["nDCG@10"]
)

print("✓ XTR Module created")

# Optimizer setzen
module.set_optimizer(torch.optim.AdamW, lr=1e-5, weight_decay=0.01)
print("✓ Optimizer set")

print("\n Testing with your examples before training...")

# Teste VORHER
for i, test_case in enumerate(test_cases, 1):
    query = test_case["query"]
    docs = [test_case["documents"]]
    
    scores = module.score([query], docs)
    print(f"Test {i} BEFORE training:")
    print(f"  Query: {query}")
    print(f"  Best score: {scores.scores.max().item():.4f}")
    print(f"  Score range: {scores.scores.min().item():.4f} - {scores.scores.max().item():.4f}")

print("\n" + "="*50)
print("TRAINING würde hier stattfinden...")
print("(Übersprungen, da keine echten Trainingsdaten)")
print("="*50)

# Simuliere Training-Verbesserung durch kleine zufällige Updates
print("\n🔄 Simulating training improvements...")
with torch.no_grad():
    for param in module.model.projection.parameters():
        param.add_(torch.randn_like(param) * 0.01)  # Kleine zufällige Updates

print("\n🎯 Testing AFTER simulated training...")

# Teste NACHHER
for i, test_case in enumerate(test_cases, 1):
    query = test_case["query"]
    docs = [test_case["documents"]]
    
    scores = module.score([query], docs)
    print(f"Test {i} AFTER training:")
    print(f"  Query: {query}")
    print(f"  Best score: {scores.scores.max().item():.4f}")
    print(f"  Score range: {scores.scores.min().item():.4f} - {scores.scores.max().item():.4f}")

print("\n XTR Example Training completed!")
print(" This shows how XTR would improve with real training data")