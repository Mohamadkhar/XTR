from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..base.searcher import ExactSearchConfig, ExactSearcher
from ...bi_encoder.bi_encoder_model import BiEncoderEmbedding
from ..base.searcher import ExactSearchConfig, ExactSearcher
if TYPE_CHECKING:
    from ...bi_encoder import BiEncoderModule, BiEncoderEmbedding


class XTRSearchConfig(ExactSearchConfig):
    def __init__(self, k: int = 100, k_train: int = None, k_inference: int = None):
        super().__init__(k=k)
        self.k_train = k_train or k
        self.k_inference = k_inference or (k_train or k)



class XTRSearcher(ExactSearcher):
    def __init__(
        self,
        index_dir: Path,
        search_config: XTRSearchConfig,
        module: BiEncoderModule,
        use_gpu: bool = True,
    ) -> None:
        super().__init__(index_dir, search_config, module, use_gpu)
        self.search_config: XTRSearchConfig
        
        # XTR-spezifische Index laden
        self.index = self._load_xtr_index()
    
    def _load_xtr_index(self):
        """Lädt den XTR-spezifischen Index (ähnlich zu TorchDenseSearcher)."""
        embeddings = torch.load(self.index_dir / "embeddings.pt", weights_only=True)
        scoring_mask = torch.load(self.index_dir / "scoring_mask.pt", weights_only=True)
        
        return {
            'embeddings': embeddings.to(self.device),
            'scoring_mask': scoring_mask.to(self.device)
        }
    
    def _score(self, query_embeddings: BiEncoderEmbedding) -> torch.Tensor:
        """XTR-spezifisches Scoring mit Token Retrieval Mask."""
        
        # XTR-Model aus dem Module extrahieren
        xtr_model = self.module.model
        if not hasattr(xtr_model, 'compute_token_retrieval_mask'):
            raise ValueError("Module ist kein XTR-Model!")
        
        # Document-Embeddings aus Index laden
        doc_embeddings = self._gather_doc_embeddings()
        
        # XTR-Scoring verwenden (nutzt Ihre score() Implementierung!)
        return xtr_model.score(query_embeddings, doc_embeddings)
    
    def _gather_doc_embeddings(self) -> BiEncoderEmbedding:
        """Lädt alle Document-Embeddings aus dem XTR-Index."""
        
        # Basierend auf TorchDenseSearcher Pattern aus #codebase
        embeddings = self.index['embeddings']  # [num_docs*doc_len, embed_dim]
        scoring_mask = self.index['scoring_mask']  # [num_docs*doc_len]
        
        # Reshape zu Multi-Vector Format
        doc_embeddings = embeddings.view(self.num_docs, -1, embeddings.size(-1))
        doc_scoring_mask = scoring_mask.view(self.num_docs, -1)
        
        # BiEncoderEmbedding erstellen
        return BiEncoderEmbedding(
            embeddings=doc_embeddings,
            scoring_mask=doc_scoring_mask,
            encoding=None  
        )