# lightning_ir/models/xtr.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Literal, Sequence, Optional, Tuple

import torch
from torch.nn import functional as F
from transformers import BatchEncoding

from lightning_ir.bi_encoder.bi_encoder_model import BiEncoderEmbedding
from ..bi_encoder import (
    BiEncoderTokenizer,
    MultiVectorBiEncoderConfig,
    MultiVectorBiEncoderModel,
)


class XTRConfig(MultiVectorBiEncoderConfig):
    r"""
    XTR (Cross/ConteXtualized Token Retrieval) — Lee et al., NeurIPS 2023.

    Training:   f_XTR   (Eq. 2) → max über Doc-Tokens, Z-Norm über Query-Tokens mit Treffern.
    Inferenz:   f_XTR′  (Eq. 4) → Missing-Similarity-Imputation m_i, anschließend 1/n über Query-Tokens.

    Für cosine wird die L2-Norm zur Similarity-Zeit erzwungen (zentral).
    """
    model_type = "xtr"

    def __init__(
        self,
        # Standard
        query_length: int = 32,
        doc_length: int = 512,
        similarity_function: Literal["cosine", "dot"] = "cosine",
        normalize: bool = False,  # für cosine ignoriert (Normierung erfolgt zentral)
        add_marker_tokens: bool = False,
        query_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,
        doc_mask_scoring_tokens: Sequence[str] | Literal["punctuation"] | None = None,

        # (API-Kompat.) – XTR nutzt intern max→mean; Felder bleiben für Downstream-Kompat.
        query_aggregation_function: Literal["sum", "mean", "max", "harmonic_mean"] = "mean",
        doc_aggregation_function: Literal["sum", "mean", "max", "harmonic_mean"] = "max",

        # Projektion (Backbone hidden_size → embedding_dim)
        embedding_dim: int = 128,
        projection: Literal["linear", "linear_no_bias"] = "linear",

        # Expansion (optional)
        query_expansion: bool = False,
        attend_to_query_expanded_tokens: bool = False,
        doc_expansion: bool = False,
        attend_to_doc_expanded_tokens: bool = False,

        # XTR-spezifisch
        k_train: int = 64,              # Top-k im Training (In-Batch)
        k_inference: int = 100,         # k′ in der Inferenz
        strict_top_k: bool = True,      # exakt k; False: Ties >= Schwelle
        missing_similarity_imputation: bool = True,  # f_XTR′ aktivieren

        # Performance
        chunk_size: int = 2048,         # für chunked Top-k
        apply_final_mask: bool = True,  # letzte 4D-Sicherheitsmaske anwenden
        **kwargs,
    ):
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
        # XTR
        self.k_train = int(k_train)
        self.k_inference = int(k_inference)
        self.strict_top_k = bool(strict_top_k)
        self.missing_similarity_imputation = bool(missing_similarity_imputation)

        # Performance
        self.chunk_size = int(chunk_size)
        self.apply_final_mask = bool(apply_final_mask)

        # Projektion & Expansion
        self.embedding_dim = int(embedding_dim)
        self.projection = projection
        self.query_expansion = bool(query_expansion)
        self.attend_to_query_expanded_tokens = bool(attend_to_query_expanded_tokens)
        self.doc_expansion = bool(doc_expansion)
        self.attend_to_doc_expanded_tokens = bool(attend_to_doc_expanded_tokens)

        # gespiegelt aus Basisklasse
        self.normalize = bool(normalize)
        self.add_marker_tokens = bool(add_marker_tokens)


class XTRModel(MultiVectorBiEncoderModel):
    config_class = XTRConfig

    def __init__(self, config: XTRConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        if config.embedding_dim is None:
            raise ValueError("Embedding dimension must be specified in the configuration.")
        use_bias = (config.projection != "linear_no_bias")
        self.projection = torch.nn.Linear(config.hidden_size, config.embedding_dim, bias=use_bias)

    # ------------------- Encoding & Masking -------------------

    def scoring_mask(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> torch.Tensor:
        input_ids = encoding["input_ids"]
        attention_mask = encoding.get("attention_mask", None)
        scoring_mask = attention_mask
        # Expansion → alles werten (oder wenn attention_mask fehlt)
        if getattr(self.config, f"{input_type}_expansion") or scoring_mask is None:
            scoring_mask = torch.ones_like(input_ids, dtype=torch.bool)
        scoring_mask = scoring_mask.bool()

        # bestimmte Token-IDs (z. B. [CLS]/[SEP]/Marker) ignorieren
        mask_scoring_input_ids = getattr(self, f"{input_type}_mask_scoring_input_ids")
        if mask_scoring_input_ids is not None:
            ignore_mask = input_ids[..., None].eq(mask_scoring_input_ids.to(input_ids.device)).any(-1)
            scoring_mask = scoring_mask & ~ignore_mask
        return scoring_mask

    def encode(self, encoding: BatchEncoding, input_type: Literal["query", "doc"]) -> BiEncoderEmbedding:
        emb = self._backbone_forward(**encoding).last_hidden_state     # [B, L, hidden]
        emb = self.projection(emb)                                     # [B, L, dim]
        # keine L2-Norm hier! (erfolgt zentral zur Similarity-Zeit)
        scoring_mask = self.scoring_mask(encoding, input_type).bool()
        assert scoring_mask.shape[:2] == encoding["input_ids"].shape[:2], "Mask shape mismatch!"
        return BiEncoderEmbedding(emb, scoring_mask, encoding)

    # ------------------- Similarity-Hilfen -------------------

    def _maybe_normalize_for_similarity(
        self, q: torch.Tensor, d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.similarity_function == "cosine":
            q = F.normalize(q, dim=-1)
            d = F.normalize(d, dim=-1)
        return q, d

    # ------------------- Token-Retrieval (Top-k) -------------------

    def compute_token_retrieval_mask(
        self,
        query_embeddings: torch.Tensor,                    # [Q, q_len, dim]
        doc_embeddings: torch.Tensor,                      # [D, d_len, dim]
        scoring_mask_query: Optional[torch.Tensor] = None, # [Q, q_len]
        scoring_mask_doc: Optional[torch.Tensor] = None,   # [D, d_len]
        k: Optional[int] = None,
        num_docs: Optional[int | Sequence[int]] = None,    # Docs pro Query (int oder Sequence)
    ) -> torch.Tensor:
        """
        Erzeuge binäre Auswahlmaske Â (Eq. 2/4).
        **CRITICAL FIX:** Query-Document-Isolation im Batch-Modus.
        Rückgabeform: [Q, q_len, D, d_len]  (True = ausgewählt).
        """
        if k is None:
            k = self.config.k_train if self.training else self.config.k_inference

        # Für cosine ggf. hier normieren (idempotent, falls bereits normiert).
        q, d = self._maybe_normalize_for_similarity(query_embeddings, doc_embeddings)

        if num_docs is not None:
            return self._compute_mask_with_doc_isolation(
                q, d, scoring_mask_query, scoring_mask_doc, int(k), num_docs
            )
        else:
            # Globale Matrix (Training mit In-Batch-Negatives)
            return self._compute_mask_global(
                q, d, scoring_mask_query, scoring_mask_doc, int(k)
            )

    def _compute_mask_with_doc_isolation(
        self,
        query_embeddings: torch.Tensor,                    # [Q, q_len, dim]
        doc_embeddings: torch.Tensor,                      # [D, d_len, dim]
        scoring_mask_query: Optional[torch.Tensor],        # [Q, q_len]
        scoring_mask_doc: Optional[torch.Tensor],          # [D, d_len]
        k: int,
        num_docs: int | Sequence[int],
    ) -> torch.Tensor:
        """
        Jede Query verarbeitet nur ihre zugehörigen Docs.
        Annahme: docs sind als [q1_docs..., q2_docs..., ...] arrangiert.
        """
        Q, q_len, dim = query_embeddings.shape
        D, d_len, _ = doc_embeddings.shape

        expected_D = (Q * num_docs) if isinstance(num_docs, int) else sum(num_docs)
        if D != expected_D:
            # Fallback, falls Anordnung nicht stimmt.
            return self._compute_mask_global(query_embeddings, doc_embeddings,
                                             scoring_mask_query, scoring_mask_doc, k)

        mask = torch.zeros(Q, q_len, D, d_len, dtype=torch.bool, device=query_embeddings.device)

        chunk = int(getattr(self.config, "chunk_size", 2048))
        strict = bool(getattr(self.config, "strict_top_k", True))

        doc_offset = 0
        for q_idx in range(Q):
            docs_for_q = (num_docs if isinstance(num_docs, int) else int(num_docs[q_idx]))
            doc_start = doc_offset
            doc_end = doc_start + docs_for_q

            q_single = query_embeddings[q_idx:q_idx+1]            # [1, q_len, dim]
            d_subset = doc_embeddings[doc_start:doc_end]          # [docs_for_q, d_len, dim]
            q_mask_single = scoring_mask_query[q_idx:q_idx+1] if scoring_mask_query is not None else None
            d_mask_subset = scoring_mask_doc[doc_start:doc_end]   if scoring_mask_doc   is not None else None

            mask_single = self._compute_single_query_mask(
                q_single, d_subset, q_mask_single, d_mask_subset, k, chunk, strict
            )  # [1, q_len, docs_for_q, d_len]

            mask[q_idx:q_idx+1, :, doc_start:doc_end, :] = mask_single
            doc_offset += docs_for_q

        return mask

    def _compute_single_query_mask(
        self,
        q_single: torch.Tensor,                            # [1, q_len, dim]
        d_subset: torch.Tensor,                            # [num_docs, d_len, dim]
        q_mask: Optional[torch.Tensor],                    # [1, q_len]
        d_mask: Optional[torch.Tensor],                    # [num_docs, d_len]
        k: int,
        chunk: int,
        strict: bool,
    ) -> torch.Tensor:
        """Top-k-Maske für eine einzelne Query über ihre Dokumente."""
        _, q_len, dim = q_single.shape
        num_docs, d_len, _ = d_subset.shape
        total_doc_tokens = num_docs * d_len
        k = max(1, min(int(k), total_doc_tokens))

        q_flat = q_single.view(-1, dim)                # [q_len, dim]
        d_flat_t = d_subset.view(-1, dim).t()          # [dim, num_docs*d_len]
        q_mask_flat = q_mask.view(-1).bool() if q_mask is not None else None
        d_mask_flat = d_mask.view(-1).bool() if d_mask is not None else None

        mask_flat = torch.zeros(q_len, total_doc_tokens, dtype=torch.bool, device=q_single.device)

        for start in range(0, q_len, chunk):
            end = min(start + chunk, q_len)
            S = q_flat[start:end] @ d_flat_t           # [chunk_q, num_docs*d_len]

            if d_mask_flat is not None:
                S[:, ~d_mask_flat] = float("-inf")
            if q_mask_flat is not None:
                S[~q_mask_flat[start:end]] = float("-inf")

            if torch.isneginf(S).all():
                continue

            eff_k = min(k, int(torch.isfinite(S).any(dim=0).sum().item()))
            if eff_k < 1:
                continue

            if strict:
                vals, idx = S.topk(eff_k, dim=1)
                finite = torch.isfinite(vals)
                if finite.any():
                    rows, _ = torch.where(finite)
                    mask_flat[rows + start, idx[finite]] = True
            else:
                topk_vals, _ = S.topk(eff_k, dim=1)
                thr = topk_vals[:, -1:].expand_as(S)
                valid = torch.isfinite(S) & (S >= thr)
                mask_flat[start:end] |= valid

        mask_4d = mask_flat.view(1, q_len, num_docs, d_len)

        if self.config.apply_final_mask:
            if q_mask is not None:
                mask_4d &= q_mask.unsqueeze(2).unsqueeze(3).bool()
            if d_mask is not None:
                mask_4d &= d_mask.unsqueeze(0).unsqueeze(1).bool()

        return mask_4d

    def _compute_mask_global(
        self,
        query_embeddings: torch.Tensor,                    # [Q, q_len, dim]
        doc_embeddings: torch.Tensor,                      # [D, d_len, dim]
        scoring_mask_query: Optional[torch.Tensor],        # [Q, q_len]
        scoring_mask_doc: Optional[torch.Tensor],          # [D, d_len]
        k: int,
    ) -> torch.Tensor:
        """Globale Q×D-Matrix (Training mit In-Batch-Negatives)."""
        Q, q_len, dim = query_embeddings.shape
        D, d_len, _ = doc_embeddings.shape
        total_doc_tokens = D * d_len
        k = max(1, min(int(k), total_doc_tokens))

        q_flat = query_embeddings.view(-1, dim)        # [Q*q_len, dim]
        d_flat_t = doc_embeddings.view(-1, dim).t()    # [dim, D*d_len]
        q_mask_flat = scoring_mask_query.view(-1).bool() if scoring_mask_query is not None else None
        d_mask_flat = scoring_mask_doc.view(-1).bool()  if scoring_mask_doc   is not None else None

        mask_flat = torch.zeros(q_flat.size(0), D * d_len, dtype=torch.bool, device=query_embeddings.device)

        chunk = int(getattr(self.config, "chunk_size", 2048))
        strict = bool(getattr(self.config, "strict_top_k", True))

        for start in range(0, q_flat.size(0), chunk):
            end = min(start + chunk, q_flat.size(0))
            S = q_flat[start:end] @ d_flat_t           # [c, D*d_len]

            if d_mask_flat is not None:
                S[:, ~d_mask_flat] = float("-inf")
            if q_mask_flat is not None:
                S[~q_mask_flat[start:end]] = float("-inf")

            if torch.isneginf(S).all():
                continue

            eff_k = min(k, int(torch.isfinite(S).any(dim=0).sum().item()))
            if eff_k < 1:
                continue

            if strict:
                vals, idx = S.topk(eff_k, dim=1)
                finite = torch.isfinite(vals)
                if finite.any():
                    rows, _ = torch.where(finite)
                    mask_flat[rows + start, idx[finite]] = True
            else:
                topk_vals, _ = S.topk(eff_k, dim=1)
                thr = topk_vals[:, -1:].expand_as(S)
                valid = torch.isfinite(S) & (S >= thr)
                mask_flat[start:end] |= valid

        mask = mask_flat.view(Q, q_len, D, d_len)

        if self.config.apply_final_mask:
            if scoring_mask_query is not None:
                mask &= scoring_mask_query.unsqueeze(2).unsqueeze(3).bool()
            if scoring_mask_doc is not None:
                mask &= scoring_mask_doc.unsqueeze(0).unsqueeze(1).bool()

        return mask

    # ------------------- Scoring -------------------

    def score(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
        num_docs: Optional[int | Sequence[int]] = None,
    ) -> torch.Tensor:
        """
        Training: f_XTR (Eq. 2)  → Z-Norm (nur Query-Tokens mit Treffern zählen).
        Inferenz (ohne Imputation): 1/n über gültige Query-Tokens.
        """
        if (not self.training) and self.config.missing_similarity_imputation:
            return self.score_with_imputation(query_embeddings, doc_embeddings, num_docs=num_docs)

        q_e = query_embeddings.embeddings            # [Q, q_len, dim]
        d_e = doc_embeddings.embeddings              # [D, d_len, dim]
        q_e, d_e = self._maybe_normalize_for_similarity(q_e, d_e)

        k = self.config.k_train if self.training else self.config.k_inference
        A = self.compute_token_retrieval_mask(
            q_e, d_e,
            scoring_mask_query=query_embeddings.scoring_mask,
            scoring_mask_doc=doc_embeddings.scoring_mask,
            k=k,
            num_docs=num_docs,  # Query-Document-Isolation in der Inferenz
        )  # [Q, q_len, D, d_len]

        S = torch.einsum("qie,dje->qidj", q_e, d_e)        # [Q, q_len, D, d_len]
        S_masked = S.masked_fill(~A, float("-inf"))

        s_qid = S_masked.max(dim=3).values                 # [Q, q_len, D]
        has_hit_qid = A.any(dim=3)                         # [Q, q_len, D]
        s_qid = torch.where(has_hit_qid, s_qid, torch.zeros_like(s_qid))

        if self.training:
            Z_qD = has_hit_qid.sum(dim=1).clamp(min=1)     # [Q, D]
            scores = s_qid.sum(dim=1) / Z_qD               # 1/Z * sum_i max_j(.)
        else:
            n_q = query_embeddings.scoring_mask.sum(dim=1).clamp(min=1)  # [Q]
            scores = s_qid.sum(dim=1) / n_q.unsqueeze(-1)                # 1/n * sum_i max_j(.)

        scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
        return scores  # [Q, D]

    def score_with_imputation(
        self,
        query_embeddings: BiEncoderEmbedding,
        doc_embeddings: BiEncoderEmbedding,
        mi_per_query: Optional[torch.Tensor] = None,   # optional [Q, q_len]
        num_docs: Optional[int | Sequence[int]] = None,
    ) -> torch.Tensor:
        """
        f_XTR′ (Eq. 4):
          score(Q, D̂) = (1/n) * sum_i max( max_{j ∈ D̂} [Â_ij · (q_i^T d_j)], m_i )
        m_i idealerweise k′-ter Wert je Query-Token; hier Fallback: k′-ter Wert über alle Kandidaten-Tokens.
        """
        q_e = query_embeddings.embeddings
        d_e = doc_embeddings.embeddings
        q_e, d_e = self._maybe_normalize_for_similarity(q_e, d_e)

        Ahat = self.compute_token_retrieval_mask(
            q_e, d_e,
            scoring_mask_query=query_embeddings.scoring_mask,
            scoring_mask_doc=doc_embeddings.scoring_mask,
            k=self.config.k_inference,
            num_docs=num_docs,
        )  # [Q, q_len, D, d_len]

        S = torch.einsum("qie,dje->qidj", q_e, d_e)        # [Q, q_len, D, d_len]
        S_masked = S.masked_fill(~Ahat, float("-inf"))
        max_retrieved = S_masked.max(dim=3).values         # [Q, q_len, D]
        has_hit_qid = Ahat.any(dim=3)
        max_retrieved = torch.where(has_hit_qid, max_retrieved, torch.full_like(max_retrieved, float("-inf")))

        if mi_per_query is None:
            S_all = S.view(S.size(0), S.size(1), -1)       # [Q, q_len, D*d_len]
            k = min(self.config.k_inference, S_all.size(-1)) if S_all.numel() > 0 else 0
            if k > 0:
                topk_vals, _ = S_all.topk(k, dim=-1)
                mi = topk_vals[..., -1]                    # [Q, q_len]
                mi = torch.where(torch.isfinite(mi), mi, torch.zeros_like(mi))
            else:
                mi = torch.zeros(S.size(0), S.size(1), device=S.device, dtype=S.dtype)
        else:
            mi = mi_per_query

        s_qid = torch.maximum(max_retrieved, mi.unsqueeze(-1))  # [Q, q_len, D]

        n_q = query_embeddings.scoring_mask.sum(dim=1).clamp(min=1)      # [Q]
        scores = s_qid.sum(dim=1) / n_q.unsqueeze(-1)                    # [Q, D]
        scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
        return scores


# ------------------- Tokenizer -------------------

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
        # PAD → MASK (Fallback auf PAD, falls MASK nicht existiert)
        input_ids = encoding["input_ids"]
        mask_id = getattr(self, "mask_token_id", None)
        if mask_id is None:
            mask_id = self.pad_token_id
        input_ids[input_ids == self.pad_token_id] = mask_id
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
