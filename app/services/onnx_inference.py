import logging
from typing import Optional
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForTokenClassification, ORTModelForSequenceClassification
from app.config import get_settings
from app.models.schemas import Entity, ClassificationResult

logger = logging.getLogger(__name__)
settings = get_settings()


class ONNXInferenceService:
    """Runs NER and classification using ONNX-optimized models for fast inference."""

    def __init__(self):
        self._ner_pipeline: Optional[pipeline] = None
        self._classification_pipeline: Optional[pipeline] = None
        self._initialized = False

    def initialize(self):
        """Lazy-load models to avoid slow startup."""
        if self._initialized:
            return

        try:
            logger.info("Loading ONNX NER model...")
            ner_model = ORTModelForTokenClassification.from_pretrained(
                settings.onnx_ner_model, export=True
            )
            ner_tokenizer = AutoTokenizer.from_pretrained(settings.onnx_ner_model)
            self._ner_pipeline = pipeline(
                "ner",
                model=ner_model,
                tokenizer=ner_tokenizer,
                aggregation_strategy="simple",
            )

            logger.info("Loading ONNX classification model...")
            cls_model = ORTModelForSequenceClassification.from_pretrained(
                settings.onnx_classification_model, export=True
            )
            cls_tokenizer = AutoTokenizer.from_pretrained(settings.onnx_classification_model)
            self._classification_pipeline = pipeline(
                "zero-shot-classification",
                model=cls_model,
                tokenizer=cls_tokenizer,
            )

            self._initialized = True
            logger.info("ONNX models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load ONNX models: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def extract_entities(self, text: str, max_length: int = 5000) -> list[Entity]:
        """Extract named entities from text using ONNX-optimized BERT-NER."""
        self.initialize()

        # Truncate for NER (model has token limit)
        truncated = text[:max_length]

        try:
            raw_entities = self._ner_pipeline(truncated)
            entities = []
            seen = set()

            for ent in raw_entities:
                key = (ent["word"].strip(), ent["entity_group"])
                if key not in seen and ent["score"] > 0.7:
                    seen.add(key)
                    entities.append(
                        Entity(
                            text=ent["word"].strip(),
                            label=ent["entity_group"],
                            confidence=round(ent["score"], 4),
                        )
                    )

            logger.info(f"Extracted {len(entities)} unique entities")
            return entities

        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return []

    def classify_text(
        self,
        text: str,
        candidate_labels: list[str] = None,
        max_length: int = 2000,
    ) -> list[ClassificationResult]:
        """Classify text into categories using zero-shot ONNX model."""
        self.initialize()

        if candidate_labels is None:
            candidate_labels = [
                "legal document",
                "financial report",
                "technical documentation",
                "research paper",
                "business correspondence",
                "news article",
                "medical document",
                "educational material",
            ]

        truncated = text[:max_length]

        try:
            result = self._classification_pipeline(
                truncated, candidate_labels=candidate_labels, multi_label=True
            )
            classifications = [
                ClassificationResult(
                    label=label,
                    confidence=round(score, 4),
                )
                for label, score in zip(result["labels"], result["scores"])
                if score > 0.3
            ]

            logger.info(f"Classification results: {[c.label for c in classifications[:3]]}")
            return classifications

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return []
