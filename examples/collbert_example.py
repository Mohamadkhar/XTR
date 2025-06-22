from lightning_ir import BiEncoderModule
from lightning_ir.models.col import ColModel

# first install lightning-ir `https://lightning-ir.webis.de/`

# Define the model
model = ColModel.from_pretrained("webis/colbert")
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

print (module.score(queries=queries, docs=documents))