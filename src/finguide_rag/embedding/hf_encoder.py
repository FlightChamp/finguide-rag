"""sentence-transformers 없이 E5 임베딩을 만드는 대체 인코더.

왜 필요한가
---------
2026-10-14, Windows 의 애플리케이션 제어 정책이 scipy 의 DLL(_tools)을 차단해
sentence-transformers import 가 실패했다. sentence-transformers 5.x 는 import 시점에
sklearn → scipy 를 끌어오는데, 임베딩 계산 자체에는 scipy 가 필요 없다.
그래서 transformers + torch 로 같은 계산(mean pooling → L2 정규화)을 직접 한다.

동일성 보장
----------
multilingual-e5 의 sentence-transformers 설정은 mean pooling + 정규화다. 이 클래스는 같은
계산을 하지만, '같다고 가정'하지 않는다. 사용 전에 기존 FAISS 인덱스에 저장된 청크 벡터를
다시 계산해 코사인 유사도로 비교하는 검사(check_against_index)를 통과해야만 쓴다.
"""

from __future__ import annotations

import numpy as np


class TransformersEncoder:
    """SentenceTransformer.encode 와 같은 호출 모양을 제공한다 (Embedder 가 그대로 쓴다)."""

    def __init__(self, model_id: str, max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self.max_length = max_length

    def encode(self, sentences, batch_size: int = 16, show_progress_bar: bool = False,
               normalize_embeddings: bool = True, convert_to_numpy: bool = True):
        out = []
        torch = self.torch
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                if normalize_embeddings:
                    vec = torch.nn.functional.normalize(vec, p=2, dim=1)
                out.append(vec.cpu().numpy())
        return np.vstack(out).astype(np.float32)


def check_against_index(embedder, faiss_store, n: int = 20, seed: int = 7) -> dict:
    """인덱스에 저장된 청크 벡터와 지금 인코더의 출력이 같은지 확인한다.

    11번 스크립트가 인덱스를 만들 때와 같은 텍스트(문서명 + 구간명 + 본문)를 다시 인코딩한다.
    반환: {"n", "min_cos", "mean_cos"}
    """
    import random

    rng = random.Random(seed)
    idx = rng.sample(range(len(faiss_store.metas)), min(n, len(faiss_store.metas)))
    texts = []
    for i in idx:
        m = faiss_store.metas[i]
        parts = [m.get("doc_display_name", ""), m.get("section", "")]
        prefix = " ".join(p for p in parts if p and p != "FAQ")
        texts.append(f"{prefix}\n{m.get('text', '')}" if prefix else m.get("text", ""))
    now = embedder.encode_passages(texts, show_progress=False)
    ref = np.vstack([faiss_store._index.reconstruct(i) for i in idx])
    cos = (now * ref).sum(1)
    return {"n": len(idx), "min_cos": float(cos.min()), "mean_cos": float(cos.mean())}
