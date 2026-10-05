"""全局路径与配置。"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "processed" / "credit.duckdb"
KB_DIR = ROOT / "knowledge_base"
ARTIFACTS = ROOT / "artifacts"
RAG_DIR = ARTIFACTS / "rag"
MODEL_PATH = ARTIFACTS / "xgb_model.json"

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
EMBED_CACHE = ARTIFACTS / "embed_cache"

LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")
LLM_EFFORT = os.getenv("LLM_EFFORT", "medium")
