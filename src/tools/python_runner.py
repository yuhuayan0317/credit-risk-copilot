"""
受限 Python 执行器：在独立子进程中运行 LLM 生成的分析代码。

安全边界：
  - 子进程 + 超时，避免死循环拖垮主进程
  - 数据库以只读方式打开，禁止外部文件/网络访问；代码里自行调用 duckdb.connect() 也只会拿到这个只读连接
  - AST 检查：只允许导入白名单模块，禁止 open/exec/eval 等内置函数
这是本地 demo 级别的隔离，不是生产级沙箱。
"""
import ast
import subprocess
import sys
import textwrap

from config import DB_PATH, ROOT

TIMEOUT_S = 60
MAX_OUTPUT = 6000
ALLOWED_IMPORTS = {"pandas", "numpy", "scipy", "math", "statistics", "json", "collections",
                   "itertools", "datetime", "re", "sklearn", "functools", "duckdb"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals",
                   "locals", "vars", "breakpoint", "exit", "quit"}

PRELUDE = textwrap.dedent(f"""
    import warnings; warnings.filterwarnings("ignore")
    import duckdb, numpy as np, pandas as pd
    from scipy import stats
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    con = duckdb.connect({str(DB_PATH)!r}, read_only=True, config={{"enable_external_access": False}})
    def sql(q):
        return con.execute(q).df()
    # 模型经常自己写 duckdb.connect(...)：不管传什么参数，都返回同一个只读连接
    duckdb.connect = lambda *args, **kwargs: con
    del duckdb
""")


class UnsafeCodeError(ValueError):
    pass


def check_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"语法错误：{e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        else:
            mods = []
        for mod in mods:
            if mod not in ALLOWED_IMPORTS:
                raise UnsafeCodeError(f"不允许导入模块 {mod}，可用：{sorted(ALLOWED_IMPORTS)}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"不允许使用 {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCodeError("不允许访问双下划线属性")


def run_python(code: str) -> tuple[bool, str]:
    check_code(code)
    try:
        proc = subprocess.run([sys.executable, "-c", PRELUDE + "\n" + code],
                              capture_output=True, text=True, timeout=TIMEOUT_S, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return False, f"执行超时（> {TIMEOUT_S} 秒），请减少计算量。"
    out = proc.stdout
    if proc.returncode != 0:
        # 只保留 traceback 的最后几行，足够定位问题
        err = "\n".join(proc.stderr.strip().splitlines()[-8:])
        return False, (out[-2000:] + "\n" if out else "") + f"执行报错：\n{err}"
    if not out.strip():
        out = "（代码执行成功，但没有输出。请用 print() 输出需要的结果。）"
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + f"\n...（输出过长，已截断，共 {len(out)} 字符）"
    return True, out
