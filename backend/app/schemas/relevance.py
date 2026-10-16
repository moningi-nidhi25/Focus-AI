from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class PageMetadata(BaseModel):
    """Extra, already-cleaned page facts the extension can see (used for YouTube videos)."""
    video_id: Optional[str] = None
    description: Optional[str] = Field(default=None, max_length=5000)
    channel: Optional[str] = Field(default=None, max_length=200)
    keywords: List[str] = Field(default_factory=list, max_length=30)  # e.g. hashtags shown on the page


class AnalyzeRequest(BaseModel):
    """Content to evaluate against a study context.

    Provide ``session_id`` (uses that session's context) or ``topic`` for a stateless preview.
    Provide ``text`` (what the extension extracted), ``html``, or just a ``url`` to fetch.
    """
    session_id: Optional[str] = None
    topic: Optional[str] = None
    learning_objectives: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None
    html: Optional[str] = None
    metadata: Optional[PageMetadata] = None
    persist: bool = True  # store the analysis / blocked content / activity events (needs session_id)

    @model_validator(mode="after")
    def _check(self):
        if not self.session_id and not self.topic:
            raise ValueError("Provide either session_id or topic.")
        if not (self.url or self.text or self.html):
            raise ValueError("Provide at least one of url, text or html.")
        return self


class SentenceScore(BaseModel):
    text: str
    score: float = Field(description="0-100 relevance of this sentence to the study context")
    level: Literal["high", "medium", "low"]


class AnalyzeResponse(BaseModel):
    analysis_id: Optional[str] = None
    url: Optional[str] = None
    title: str = ""
    source_type: str = "webpage"
    relevance_score: float
    classification: Literal["RELEVANT", "PARTIALLY_RELEVANT", "DISTRACTING"]
    decision: Literal["ALLOW", "WARN", "BLOCK"]
    explanation: str
    reasons: List[str] = []
    matching_concepts: List[str] = []
    highlighted_sentences: List[SentenceScore] = []
    heatmap: List[SentenceScore] = []
    signals: Dict[str, Optional[float]] = {}
    explanation_method: str
    embedding_backend: str
    llm_used: bool = False
    low_content: bool = False
    insufficient_content: bool = False
    word_count: int = 0
    # How the score was produced and what evidence was available (useful for debugging/explanations)
    evaluation_path: str = "embedding"  # embedding | embedding+llm | llm+embedding | insufficient
    llm_status: str = "not_configured"  # not_configured | not_needed | ok | failed:<kind>
    content_quality: str = "sufficient"  # sufficient | low | insufficient
    content_sources: Dict[str, Any] = {}


class OverrideRequest(BaseModel):
    session_id: str
    url: str
    title: Optional[str] = None


class HistoryItem(BaseModel):
    id: str
    url: str
    title: Optional[str] = None
    relevance_score: float
    classification: Optional[str] = None
    decision: str
    analyzed_at: datetime


class EngineInfo(BaseModel):
    embedding_backend: str
    dense_embeddings: bool
    llm_enabled: bool
    llm_provider: Optional[str] = None  # never includes the key
    llm_model: Optional[str] = None
    allow_threshold: float
    warn_threshold: float
    explanation_method: str


# ---------------------------------------------------------------- token-level (Integrated Gradients) explanation
class ExplainTokensRequest(BaseModel):
    """Ask for word-level Integrated Gradients of ONE sentence against the study context.

    Provide ``session_id`` (uses that session's context) or ``topic`` (+ optional objectives).
    """
    session_id: Optional[str] = None
    topic: Optional[str] = Field(default=None, max_length=500)
    learning_objectives: Optional[str] = Field(default=None, max_length=5000)
    sentence: str = Field(min_length=1, max_length=2000)
    steps: Optional[int] = Field(default=None, ge=8, le=64, description="IG integration steps (default 32)")

    @model_validator(mode="after")
    def _check(self):
        self.sentence = self.sentence.strip()
        if not self.sentence:
            raise ValueError("sentence must contain text.")
        if not self.session_id and not (self.topic and self.topic.strip()):
            raise ValueError("Provide either session_id or topic.")
        return self


class TokenPart(BaseModel):
    token: str
    attribution: float


class WordAttributionOut(BaseModel):
    word: str
    start: int = Field(description="character offset of the word in the explained sentence")
    end: int
    attribution: float = Field(description="RAW Integrated Gradients value (cosine units), signed")
    normalized: float = Field(description="attribution / max|attribution| in [-1, 1]; for colouring only")
    tokens: List[TokenPart] = []


class ExplainTokensResponse(BaseModel):
    available: bool
    reason: Optional[str] = None  # unsupported_backend | unsupported_model | empty_input | torch_missing | attribution_failed
    message: Optional[str] = None
    method: str = "integrated-gradients"
    target: str = ""
    scope: str = ""
    baseline: str = ""
    score_semantics: str = ""
    words: List[WordAttributionOut] = []
    n_tokens: int = 0
    n_tokens_total: int = 0
    truncated: bool = False
    steps: int = 0
    target_value: Optional[float] = Field(default=None, description="F(x): cosine(sentence, context)")
    baseline_value: Optional[float] = Field(default=None, description="F(baseline)")
    sum_attributions: Optional[float] = None
    completeness_error: Optional[float] = None
    converged: Optional[bool] = None
    embedding_backend: str = ""
    cached: bool = False
