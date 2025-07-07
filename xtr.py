from lightning_ir.bi_encoder.bi_encoder_model import BiEncoderEmbedding
from ..bi_encoder import BiEncoderTokenizer, MultiVectorBiEncoderConfig, MultiVectorBiEncoderModel
from transformers import BatchEncoding
from typing import Literal, Sequence
import torch

class XTRConfig(MultiVectorBiEncoderConfig):
    
    """Konfigurationsklasse für das XTR (Cross Token Retrieval) Modell.
    XTR ist ein effizienter Multi-Vector Information Retrieval Ansatz, der für jedes
    Query Token nur die k relevantesten Document Tokens für die Ähnlichkeitsberechnung verwendet.
    """
    model_type = "xtr"

    def __init__(
        self,
        # Standard BiEncoder Parameter
        query_length: int = 32,
        doc_length: int = 512,
        similarity_function: Literal["cosine", "dot"] = "dot",
        normalize: bool = False,
        add_marker_tokens: bool = False,
        query_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        doc_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        query_aggregation_function: Literal["sum", "mean", "max", "harmonic_mean"] = "sum",
        doc_aggregation_function: Literal["sum", "mean", "max", "harmonic_mean"] = "max",
        # Embedding Projektion
        embedding_dim: int = 128,
        projection: Literal["linear", "linear_no_bias"] = "linear",
        # Query/Doc Expansion (experimentell)
        query_expansion: bool = False,
        attend_to_query_expanded_tokens: bool = False,
        # XTR spezifische Parameter (Kern des Algorithmus)
        k_train: int = 10,      # Topk Tokens während des Trainings
        k_inference: int = 5,   # Topk Tokens während der Inferenz
        strict_top_k: bool = False,  # Exakt k Tokens vs. Schwellenwert basiert
        missing_similarity_imputation: bool = False,    # Fehlende Ähnlichkeiten ersetzen (fXTR′)
        doc_expansion: bool = False,
        attend_to_doc_expanded_tokens: bool = False,
        **kwargs,
    ):
        """Initialisiert die XTR-Konfiguration. Args:
            k_train: Anzahl der Topk Document Tokens pro QueryToken beim Training
            k_inference: Anzahl der Topk Document Tokens pro Query Token bei Inferenz
            strict_top_k: Wenn True, exakt k Tokens; wenn False, alle Tokens >= k-ter Score
            missing_similarity_imputation: Wenn True, ersetzt nicht-ausgewählte Tokens 
                                         mit dem k-ten Score (fXTR Variante) welches wir noch nicht implementiert haben.
        """
        super().__init__(
            query_length=query_length,
            doc_length=doc_length,
            similarity_function=similarity_function,
            add_marker_tokens=add_marker_tokens,
            query_mask_scoring_tokens=query_mask_scoring_tokens,
            doc_mask_scoring_tokens=doc_mask_scoring_tokens,
            query_aggregation_function=query_aggregation_function,
            doc_aggregation_function=doc_aggregation_function,
            **kwargs,
        )
        self.k_train = k_train
        self.k_inference = k_inference
        self.strict_top_k = strict_top_k
        self.missing_similarity_imputation = missing_similarity_imputation
        self.embedding_dim = embedding_dim
        self.projection = projection
        self.query_expansion = query_expansion
        self.attend_to_query_expanded_tokens = attend_to_query_expanded_tokens
        self.doc_expansion = doc_expansion
        self.attend_to_doc_expanded_tokens = attend_to_doc_expanded_tokens
        self.normalize = normalize
        self.add_marker_tokens = add_marker_tokens

class XTRModel(MultiVectorBiEncoderModel):
    config_class = XTRConfig

    def __init__(self, config: XTRConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
    
        if config.embedding_dim is None:
            raise ValueError("Embedding dimension must be specified in the configuration.")
        self.projection = torch.nn.Linear(
            config.hidden_size, config.embedding_dim, bias="no_bias" not in config.projection
        )

    def scoring_mask(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> torch.Tensor:
        """
        Erstellt eine Maske für Tokens, die beim Scoring berücksichtigt werden sollen.
        Args:
            encoding: Tokenisierte Input-Sequenz
            input_type: "query" oder "doc" 
        Returns:
            torch.Tensor: Boolean Maske [batch_size, seq_len]
        """
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        scoring_mask = attention_mask
        expansion = getattr(self.config, f"{input_type}_expansion")
        if expansion or scoring_mask is None:
            scoring_mask = torch.ones_like(input_ids, dtype=torch.bool)
        scoring_mask = scoring_mask.bool()
        mask_scoring_input_ids = getattr(self, f"{input_type}_mask_scoring_input_ids")
        if mask_scoring_input_ids is not None:
            ignore_mask = input_ids[..., None].eq(mask_scoring_input_ids.to(input_ids.device)).any(-1)
            scoring_mask = scoring_mask & ~ignore_mask
        return scoring_mask

    def encode(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> BiEncoderEmbedding:
        embeddings = self._backbone_forward(**encoding).last_hidden_state
        embeddings = self.projection(embeddings)
        if self.config.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        #scoring_mask = self.scoring_mask(encoding, input_type)
        scoring_mask = self.scoring_mask(encoding, input_type).bool()
        assert scoring_mask.shape[1] == encoding["input_ids"].shape[1], "Mask shape mismatch!"
        return BiEncoderEmbedding(embeddings, scoring_mask, encoding)

    def compute_token_retrieval_mask(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        scoring_mask_query: torch.Tensor | None = None,
        scoring_mask_doc: torch.Tensor | None = None,
        k: int = None,
    ) -> torch.Tensor:
        """
Berechnet die Token Retrieval Maske Â_ij gemäß dem XTR Algorithmus.

Für jedes Query Token q_i werden die k ähnlichsten Document-Tokens d_j ausgewählt.
Dies bildet das Herzstück des effizienten Scorings in XTR.

Ablauf:
1. Berechne die Ähnlichkeitsmatrix für alle Tokenpaare (q_i, d_j)
2. Wende optional Masken für gültige Query und Document Tokens an
3. Bestimme für jedes Query Token die Topk ähnlichsten Dokument Tokens
4. Erzeuge eine boolesche Maske, die anzeigt, welche Tokenpaare berücksichtigt werden sollen

Args:
    query_embeddings (torch.Tensor): Query Token Embeddings mit Shape [Q, q_len, d_model]
    doc_embeddings (torch.Tensor): Dokument Token Embeddings mit Shape [D, d_len, d_model]
    scoring_mask_query (torch.Tensor | None): Maskierung gültiger Query Tokens [Q, q_len]
    scoring_mask_doc (torch.Tensor | None): Maskierung gültiger Dokument Tokens [D, d_len]
    k (int | None): Anzahl Topk Tokenpaare pro Query Token. Falls None, wird `k_train` aus der Konfiguration verwendet.

Returns:
    torch.Tensor: Boolesche Maske mit Shape [Q, q_len, D, d_len], wobei True bedeutet:
                  Tokenpaar (q_i, d_j) wird im Scoring berücksichtigt (d.h. gehört zu den Topk).
"""

        if k is None:
            k = getattr(self.config, 'k_train', 10)

        Q, q_len, d_model = query_embeddings.shape
        D, d_len, _ = doc_embeddings.shape
        total_doc_tokens = D * d_len
        k = min(k, total_doc_tokens)

        query_flat = query_embeddings.view(-1, d_model)
        doc_flat = doc_embeddings.view(-1, d_model)
        similarity = torch.matmul(query_flat, doc_flat.T)
        sim_matrix = similarity.view(Q, q_len, D, d_len)

        if scoring_mask_query is not None:
            sim_matrix = sim_matrix.masked_fill(~scoring_mask_query.unsqueeze(2).unsqueeze(3), float('-inf'))
        if scoring_mask_doc is not None:
            sim_matrix = sim_matrix.masked_fill(~scoring_mask_doc.unsqueeze(0).unsqueeze(1), float('-inf'))

        sim_flat = sim_matrix.view(Q, q_len, -1)
        top_k_vals, _ = sim_flat.topk(k, dim=-1)
        threshold = top_k_vals[..., -1].unsqueeze(-1).unsqueeze(-1)
        return sim_matrix >= threshold
    
    
    def score(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
        num_docs: int = None  
    ) -> torch.Tensor:
        """
        Standard XTR-Scoring mit selektiver Token-Retrieval.
    
         - Aufgerufen von BiEncoderModule.score() bei jeder Inference/Evaluation
         - Überschreibt MultiVectorBiEncoderModel.score() - zentrale Scoring-Pipeline  
         - Implementiert XTR-Algorithmus mit Top-k Token-Selektion für Effizienz
    
        Entscheidungslogik:
         - Wenn missing_similarity_imputation=False → Standard XTR (diese Methode)
         - Wenn missing_similarity_imputation=True → score_with_imputation() (fXTR′)
        """
        if self.config.missing_similarity_imputation:
            return self.score_with_imputation(query_embeddings, doc_embeddings)

        k = getattr(self.config, 'k_inference', getattr(self.config, 'k_train', 5))
        retrieval_mask = self.compute_token_retrieval_mask(
            query_embeddings.embeddings,
            doc_embeddings.embeddings,
            scoring_mask_query=query_embeddings.scoring_mask,
            scoring_mask_doc=doc_embeddings.scoring_mask,
            k=k
        )
        # In score():
        similarity = torch.einsum('qle,dme->qlmd', query_embeddings.embeddings, doc_embeddings.embeddings)
        if query_embeddings.scoring_mask is not None:
            # Masken anwenden
            similarity = similarity.masked_fill(~query_embeddings.scoring_mask.unsqueeze(2).unsqueeze(3), 0.0)
        if doc_embeddings.scoring_mask is not None:
           # similarity = similarity.masked_fill(~doc_embeddings.scoring_mask.unsqueeze(0).unsqueeze(1), 0.0)
            
           doc_mask = doc_embeddings.scoring_mask.bool()  # [D, d_len]
           assert doc_mask.shape[0] == similarity.shape[2]
           assert doc_mask.shape[1] == similarity.shape[3]
           similarity = similarity.masked_fill(~doc_mask.unsqueeze(0).unsqueeze(1), 0.0)

        # XTR: Nur Topk Token Paare berücksichtigen
        similarity = similarity * retrieval_mask.float()
        return self._aggregate_scores(similarity)
    #diese Methode ist für die fXTR Variante mit Missing Similarity Imputation
    #diese methode ist nicht komplet fertig. deshalb wird sie nicht in meinem Beispiel aufgerufen.
    def score_with_imputation(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
        num_docs: int = None 
    ) -> torch.Tensor:
        
        """
        fXTR Variante mit Missing Similarity Imputation.
    
       - Aufgerufen von score() wenn config.missing_similarity_imputation=True
       - Alternative Scoring-Methode für experimentelle fXTR Variante
       - Ersetzt ignorierte Token-Paare mit k-tem Score statt 0 → bessere Recall
    
    Unterschied zu Standard XTR:
    - Standard: Nicht-Top-k Token-Paare = 0 (werden ignoriert)  
    - fXTR: Nicht-Top-k Token-Paare = mᵢ (k-ter höchster Score)

    """
        k = getattr(self.config, 'k_inference', getattr(self.config, 'k_train', 5))

        Q, q_len, d_model = query_embeddings.embeddings.shape
        D, d_len, _ = doc_embeddings.embeddings.shape
        total_doc_tokens = D * d_len
        k = min(k, total_doc_tokens)

        similarity = torch.einsum("qle,dme->qlmd", query_embeddings.embeddings, doc_embeddings.embeddings)

        if query_embeddings.scoring_mask is not None:
            similarity = similarity.masked_fill(~query_embeddings.scoring_mask.unsqueeze(2).unsqueeze(3), float("-inf"))
        if doc_embeddings.scoring_mask is not None:
            similarity = similarity.masked_fill(~doc_embeddings.scoring_mask.unsqueeze(0).unsqueeze(1), float("-inf"))

        sim_flat = similarity.view(Q, q_len, -1)
        topk_values, _ = sim_flat.topk(k, dim=-1)
        mi = topk_values[..., -1]  # [Q, q_len]

        threshold = mi.unsqueeze(-1).unsqueeze(-1)
        retrieval_mask = similarity >= threshold

        similarity_masked = similarity.masked_fill(~retrieval_mask, 0.0)
        similarity_imputed = similarity_masked + (~retrieval_mask).float() * mi.unsqueeze(-1).unsqueeze(-1)

        return self._aggregate_scores(similarity_imputed)
    
   #die _aggregate_scores() Methode
    def _aggregate_scores(self, similarity: torch.Tensor) -> torch.Tensor:
        """
        XTR-spezifische Aggregation mit Lightning IR Standard-Methoden.

        Aggregiert tokenweise Ähnlichkeiten:
        1. Dokumentebene (über d_len)
        2. Anfrageebene (über q_len)

        Returns:
            torch.Tensor: Finale Scores [Q, D]
        """    
        # Nutze Parent-Klasse _aggregate() Methode
        # Document-Aggregation (dim=-1)
        doc_scores = self._aggregate(
            similarity, 
            mask=None,  # Keine zusätzliche Maske nötig (bereits in similarity eingebaut)
            query_aggregation_function=self.config.doc_aggregation_function,
            dim=-1
        ).squeeze(-1)  # [Q, q_len, D]
        
        # Query-Aggregation (dim=1)  
        final_scores = self._aggregate(
            doc_scores.unsqueeze(-1),  # Für _aggregate() Format
            mask=None,
            query_aggregation_function=self.config.query_aggregation_function,
            dim=1
        ).squeeze(-1).squeeze(-1)  # [Q, D]
        
        return final_scores

class XTRTokenizer(BiEncoderTokenizer):
    config_class = XTRConfig

    def __init__(
        self,
        *args,
        query_length: int = 32,
        doc_length: int = 512,
        add_marker_tokens: bool = False,
        query_expansion: bool = False,
        attend_to_query_expanded_tokens: bool = False,
        doc_expansion: bool = False,
        attend_to_doc_expanded_tokens: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            query_length=query_length,
            doc_length=doc_length,
            query_expansion=query_expansion,
            attend_to_query_expanded_tokens=attend_to_query_expanded_tokens,
            doc_expansion=doc_expansion,
            attend_to_doc_expanded_tokens=attend_to_doc_expanded_tokens,
            add_marker_tokens=add_marker_tokens,
            **kwargs,
        )
        self.query_expansion = query_expansion
        self.attend_to_query_expanded_tokens = attend_to_query_expanded_tokens
        self.doc_expansion = doc_expansion
        self.attend_to_doc_expanded_tokens = attend_to_doc_expanded_tokens

    def _expand(self, encoding: BatchEncoding, attend_to_expanded_tokens: bool) -> BatchEncoding:
        input_ids = encoding["input_ids"]
        input_ids[input_ids == self.pad_token_id] = self.mask_token_id
        encoding["input_ids"] = input_ids
        if attend_to_expanded_tokens:
            encoding["attention_mask"].fill_(1)
        return encoding

    def tokenize_input_sequence(
        self, text: Sequence[str] | str, input_type: Literal["query", "doc"], *args, **kwargs
    ) -> BatchEncoding:
        expansion = getattr(self, f"{input_type}_expansion")
        if expansion:
            kwargs["padding"] = "max_length"
        encoding = super().tokenize_input_sequence(text, input_type, *args, **kwargs)
        if expansion:
            encoding = self._expand(encoding, getattr(self, f"attend_to_{input_type}_expanded_tokens"))
        return encoding
