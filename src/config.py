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

# LLM：配置了智谱 Key 就默认用智谱（免费），否则用 Claude；也可以用 LLM_PROVIDER 显式指定
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or ("zhipu" if ZHIPU_API_KEY else "anthropic")
_DEFAULT_MODEL = {"zhipu": "glm-4.7-flash", "anthropic": "claude-opus-5-5"}
_DEFAULT_JUDGE = {"zhipu": "glm-4.7-flash", "anthropic": "claude-sonnet-5-5"}
LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULT_MODEL.get(LLM_PROVIDER, "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or _DEFAULT_JUDGE.get(LLM_PROVIDER, "")
LLM_EFFORT = os.getenv("LLM_EFFORT", "medium")   # 仅 Claude 使用
