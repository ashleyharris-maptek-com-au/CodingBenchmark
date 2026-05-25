"""
Helpers for running JavaScript and TypeScript benchmark submissions.

JavaScript is executed with Node.js. TypeScript is compiled with the open-source
TypeScript compiler from npm when available, then executed with Node.js. On
newer Node versions that can strip TypeScript syntax directly, that runtime is
used as a fallback if no compiler is installed.
"""

import hashlib
import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

from native_compiler import describe_this_pc

SCRIPT_CACHE_DIR = Path(__file__).parent / "compile_cache" / "script_runners"
TYPESCRIPT_TOOLS_DIR = Path(__file__).parent / "compile_cache" / "typescript_tools"
_TS_INSTALL_ATTEMPTED = False
_NODE_STRIPS_TYPES: Optional[bool] = None


class ScriptExecutionError(Exception):
  pass


class ScriptCompilationError(Exception):
  pass


class ScriptRunResult:

  def __init__(self,
               success: bool,
               stdout: str = "",
               stderr: str = "",
               exec_time: float = 0.0,
               return_code: int = -1,
               error: str = "",
               error_stage: str = ""):
    self.success = success
    self.stdout = stdout
    self.stderr = stderr
    self.exec_time = exec_time
    self.return_code = return_code
    self.error = error
    self.error_stage = error_stage

  def __bool__(self):
    return self.success

  def error_message(self, max_len: int = 240) -> str:
    if self.success:
      return ""
    msg = self.error or self.stderr
    return msg[:max_len]


def _hash_text(text: str) -> str:
  return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _sanitize_name(name: str) -> str:
  return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def _ensure_dir(path: Path) -> Path:
  path.mkdir(parents=True, exist_ok=True)
  return path


def _node_path() -> Optional[str]:
  node = shutil.which("node")
  if node:
    return node
  if platform.system() == "Windows":
    candidate = r"C:\Program Files\nodejs\node.exe"
    if os.path.exists(candidate):
      return candidate
  return None


def _kill_process(process: subprocess.Popen) -> None:
  try:
    if platform.system() == "Windows":
      subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, timeout=5)
    else:
      os.killpg(os.getpgid(process.pid), signal.SIGKILL)
  except Exception:
    try:
      process.kill()
    except Exception:
      pass
  try:
    process.wait(timeout=2)
  except Exception:
    pass


def _run_node(script_path: Path, input_data: str, timeout: float) -> Tuple[str, str, float, int]:
  node = _node_path()
  if not node:
    raise ScriptExecutionError("Node.js was not found on PATH")

  start = time.time()
  process = None
  try:
    process = subprocess.Popen(
      [node, "--max-old-space-size=4096", str(script_path)],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0)
    stdout, stderr = process.communicate(input=input_data.encode("utf-8"), timeout=timeout)
    elapsed = time.time() - start
    return (stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace"),
            elapsed, process.returncode)
  except subprocess.TimeoutExpired:
    if process:
      _kill_process(process)
    raise ScriptExecutionError(f"Execution timed out after {timeout} seconds")
  except Exception as e:
    if process:
      _kill_process(process)
    raise ScriptExecutionError(f"Execution error: {e}")


def execute_javascript(code: str,
                       engine_name: str,
                       input_data: str = "",
                       timeout: float = 60) -> ScriptRunResult:
  node = _node_path()
  if not node:
    return ScriptRunResult(False, error="Node.js was not found", error_stage="runtime_missing")

  source_hash = _hash_text(code)
  cache_dir = _ensure_dir(SCRIPT_CACHE_DIR / _sanitize_name(engine_name) / "javascript")
  script_path = cache_dir / f"{source_hash}.js"
  if not script_path.exists():
    script_path.write_text(code, encoding="utf-8")

  try:
    stdout, stderr, exec_time, return_code = _run_node(script_path, input_data, timeout)
    return ScriptRunResult(return_code == 0, stdout, stderr, exec_time, return_code,
                           stderr if return_code != 0 else "",
                           "execution" if return_code != 0 else "")
  except ScriptExecutionError as e:
    return ScriptRunResult(False,
                           error=str(e),
                           error_stage="timeout" if "timed out" in str(e) else "execution")


def _local_tsc_js() -> Path:
  return TYPESCRIPT_TOOLS_DIR / "node_modules" / "typescript" / "lib" / "tsc.js"


def _ensure_local_typescript() -> Optional[Path]:
  global _TS_INSTALL_ATTEMPTED
  tsc_js = _local_tsc_js()
  if tsc_js.exists():
    return tsc_js

  if _TS_INSTALL_ATTEMPTED:
    return None
  _TS_INSTALL_ATTEMPTED = True

  npm = shutil.which("npm")
  if not npm:
    return None

  try:
    TYPESCRIPT_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([npm, "install", "--prefix",
                    str(TYPESCRIPT_TOOLS_DIR), "typescript@latest"],
                   capture_output=True,
                   text=True,
                   encoding='utf-8',
                   errors='replace',
                   timeout=180)
  except Exception:
    return None

  return tsc_js if tsc_js.exists() else None


def _find_tsc_command() -> Optional[list]:
  node = _node_path()
  local_tsc = _ensure_local_typescript()
  if node and local_tsc:
    return [node, str(local_tsc)]

  tsc = shutil.which("tsc")
  if tsc:
    return [tsc]

  return None


def _node_supports_type_stripping() -> bool:
  global _NODE_STRIPS_TYPES
  if _NODE_STRIPS_TYPES is not None:
    return _NODE_STRIPS_TYPES

  node = _node_path()
  if not node:
    _NODE_STRIPS_TYPES = False
    return _NODE_STRIPS_TYPES

  try:
    res = subprocess.run([node, "-e", "let x: number = 1; console.log(x)"],
                         capture_output=True,
                         text=True,
                         encoding='utf-8',
                         errors='replace',
                         timeout=10)
    _NODE_STRIPS_TYPES = res.returncode == 0 and res.stdout.strip() == "1"
  except Exception:
    _NODE_STRIPS_TYPES = False
  return _NODE_STRIPS_TYPES


def _compile_typescript(code: str, engine_name: str) -> Path:
  source_hash = _hash_text(code)
  cache_dir = _ensure_dir(SCRIPT_CACHE_DIR / _sanitize_name(engine_name) / "typescript")
  source_path = cache_dir / f"{source_hash}.ts"
  output_path = cache_dir / f"{source_hash}.js"

  if output_path.exists():
    return output_path
  if not source_path.exists():
    source_path.write_text(code, encoding="utf-8")

  tsc_cmd = _find_tsc_command()
  if tsc_cmd:
    cmd = tsc_cmd + [
      "--target",
      "ES2022",
      "--module",
      "CommonJS",
      "--moduleResolution",
      "node",
      "--skipLibCheck",
      "--esModuleInterop",
      "--noEmitOnError",
      "false",
      "--outDir",
      str(cache_dir),
      str(source_path),
    ]
    result = subprocess.run(cmd,
                            capture_output=True,
                            text=True,
                            encoding='utf-8',
                            errors='replace',
                            timeout=120)
    # TypeScript still emits JavaScript for many useful submissions even when
    # node type declarations are absent. Prefer runnable output over rejecting
    # harmless diagnostics about require/process/console.
    if output_path.exists():
      return output_path
    if result.returncode != 0:
      diagnostic = result.stderr or result.stdout or "Unknown TypeScript compilation error"
      raise ScriptCompilationError(diagnostic[:2000])

  if _node_supports_type_stripping():
    return source_path

  raise ScriptCompilationError(
    "No TypeScript compiler found. Install with: npm install --prefix compile_cache/typescript_tools typescript"
  )


def execute_typescript(code: str,
                       engine_name: str,
                       input_data: str = "",
                       timeout: float = 60) -> ScriptRunResult:
  node = _node_path()
  if not node:
    return ScriptRunResult(False, error="Node.js was not found", error_stage="runtime_missing")

  try:
    script_path = _compile_typescript(code, engine_name)
  except ScriptCompilationError as e:
    return ScriptRunResult(False, error=str(e), error_stage="compilation")
  except Exception as e:
    return ScriptRunResult(False, error=f"TypeScript setup failed: {e}", error_stage="compilation")

  try:
    stdout, stderr, exec_time, return_code = _run_node(script_path, input_data, timeout)
    return ScriptRunResult(return_code == 0, stdout, stderr, exec_time, return_code,
                           stderr if return_code != 0 else "",
                           "execution" if return_code != 0 else "")
  except ScriptExecutionError as e:
    return ScriptRunResult(False,
                           error=str(e),
                           error_stage="timeout" if "timed out" in str(e) else "execution")


def describe_javascript_runtime() -> str:
  node = _node_path()
  if not node:
    return "Node.js: not found"
  try:
    version = subprocess.run([node, "--version"],
                             capture_output=True,
                             text=True,
                             encoding='utf-8',
                             errors='replace',
                             timeout=10).stdout.strip()
  except Exception:
    version = "unknown"
  return f"{describe_this_pc()}\nNode.js: {version}"


def describe_typescript_runtime() -> str:
  node_desc = describe_javascript_runtime()
  tsc_cmd = _find_tsc_command()
  if not tsc_cmd:
    fallback = "yes" if _node_supports_type_stripping() else "no"
    return f"{node_desc}\nTypeScript compiler: not found\nNode TypeScript stripping fallback: {fallback}"
  try:
    version = subprocess.run(tsc_cmd + ["--version"],
                             capture_output=True,
                             text=True,
                             encoding='utf-8',
                             errors='replace',
                             timeout=20).stdout.strip()
  except Exception:
    version = "unknown"
  return f"{node_desc}\nTypeScript compiler: {version}"
