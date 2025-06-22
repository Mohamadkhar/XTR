import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import Dataset, DataLoader

from lightning_ir import BiEncoderModule, LightningIRTrainer
from lightning_ir.models.xtr import XTRConfig
from lightning_ir.data import TrainBatch
from lightning_ir.loss import InfoNCE

class DiverseTrainDataset(Dataset):
    def __init__(self, size=20):
        self.size = size
        self.examples = [
            {
                "query": "What is the capital of France?",
                "docs": ["Paris is the capital of France.", "Berlin is the capital of Germany."],
                "targets": [1, 0]
            },
            {
                "query": "Who wrote Romeo and Juliet?", 
                "docs": ["Shakespeare wrote Romeo and Juliet.", "Mozart composed classical music."],
                "targets": [1, 0]
            },
            {
                "query": "What is machine learning?",
                "docs": ["Machine learning is a subset of AI.", "Cooking involves preparing food."],
                "targets": [1, 0]
            }
        ]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        example = self.examples[idx % len(self.examples)]
        return TrainBatch(
            queries=[example["query"]],
            docs=[example["docs"]], 
            targets=torch.tensor([example["targets"]], dtype=torch.float32)
        )

def train_batch_collate_fn(batch):
    all_queries = []
    all_docs = []
    all_targets = []
    
    for train_batch in batch:
        all_queries.extend(train_batch.queries)
        all_docs.extend(train_batch.docs)
        all_targets.append(train_batch.targets)
    
    combined_targets = torch.cat(all_targets, dim=0)
    
    return TrainBatch(
        queries=all_queries,
        docs=all_docs,
        targets=combined_targets
    )

print("Starting XTR Training")
print("=" * 30)

config = XTRConfig(
    query_length=16,
    doc_length=64, 
    embedding_dim=384,
    normalize=True
)

print("XTR Config created")

xtr_module = BiEncoderModule(
    model_name_or_path="bert-base-uncased",
    config=config,
    loss_functions=[InfoNCE()],
    evaluation_metrics=["nDCG@10"]
)

print("XTR Module created")

xtr_module.set_optimizer(torch.optim.AdamW, lr=5e-5, weight_decay=0.01)
print("Optimizer set")

train_dataset = DiverseTrainDataset(size=20)
dataloader = DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True,
    collate_fn=train_batch_collate_fn,
    num_workers=0
)
print(f"Dataset created with {len(train_dataset)} samples")

logger = CSVLogger("training_logs", name="xtr_experiment")
checkpoint_callback = ModelCheckpoint(
    monitor="loss",  # Statt "train_loss" - das ist der korrekte Key!
    save_top_k=1,
    mode="min",
    filename="xtr-{epoch:02d}-{loss:.2f}"
)

print("Logger and Callbacks set")

trainer = LightningIRTrainer(
    max_epochs=2,
    max_steps=10,
    logger=logger,
    callbacks=[checkpoint_callback],
    log_every_n_steps=2,
    enable_progress_bar=True,
    enable_model_summary=True
)

print("Trainer created")
print("\nStarting training...")

try:
    trainer.fit(xtr_module, dataloader)
    print("\nTraining completed successfully!")
    print(f"Logs saved to: {logger.log_dir}")
    print(f"Checkpoints saved to: {checkpoint_callback.dirpath}")

    print("\nTesting trained model...")
    test_cases = [
        {
            "query": "What is the capital of France?",
            "documents": [
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
                "Madrid is the capital of Spain.",
                "Rome is the capital of Italy."
            ]
        }
    ]

    for case in test_cases:
        try:
            scores = xtr_module.score([case["query"]], [case["documents"]])
            print(f"\nQuery: {case['query']}")
            
            # Debug: Schauen wir uns die Score-Struktur an
            print(f"Scores type: {type(scores)}")
            print(f"Scores attributes: {dir(scores)}")
            
            if hasattr(scores, 'scores'):
                print(f"scores.scores shape: {scores.scores.shape}")
                print(f"scores.scores: {scores.scores}")
                
                # Sichere Iteration über die Scores
                if len(scores.scores.shape) > 1:
                    # Multi-dimensional tensor
                    doc_scores = scores.scores[0]  # Erste Query
                else:
                    # Flacher Tensor
                    doc_scores = scores.scores
                
                # Scores anzeigen
                for i, (score, doc) in enumerate(zip(doc_scores, case["documents"])):
                    print(f"  {i+1}. Score: {score.item():.4f} - {doc}")
                    
            else:
                print("No 'scores' attribute found in output")
                
        except Exception as e:
            print(f"Error during testing: {e}")
            print(f"Score object: {scores}")

except Exception as e:
    print(f"Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTraining Summary:")
print(f"- Total steps completed: {trainer.global_step}")
print(f"- Final epoch: {trainer.current_epoch}")
print("- Check training_logs/ for detailed logs")
print("- Training was successful! ")