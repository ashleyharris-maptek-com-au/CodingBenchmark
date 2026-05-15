"""
Common utilities for executing LLM solver code with debugger isolation.
This module provides a unified approach to running subprocesses safely,
preventing debugger interference across all test files.

Also provides streaming data infrastructure for large test cases that
don't fit in memory.
"""

import os
import re
import shutil
import sys
import time
import subprocess
import tempfile
import platform
import json
import signal
import hashlib
import threading
from pathlib import Path
from typing import Generator, Callable, Optional, Union, Any, Iterator

try:
  import numpy as np
except ImportError:
  np = None

# Directory for cached streaming data
STREAMING_CACHE_DIR = Path(tempfile.gettempdir()) / "codingbenchmark_streaming_cache"

# Directory for cached grade/report results
GRADE_CACHE_DIR = Path(tempfile.gettempdir()) / "codingbenchmark_grade_cache"

# Directory for cached baseline/reference computations
BASELINE_CACHE_DIR = Path(tempfile.gettempdir()) / "codingbenchmark_baseline_cache"


def _strip_comment_prefix(line: str) -> str:
  stripped = line.lstrip()
  if stripped.startswith("///"):
    stripped = stripped[3:]
  elif stripped.startswith("//"):
    stripped = stripped[2:]
  elif stripped.startswith("#"):
    stripped = stripped[1:]
  return stripped.lstrip()


def _split_leading_comment(text: str) -> tuple:
  """Return (comment, remainder) if a leading comment block exists."""
  block_match = re.match(r"\s*/\*([\s\S]*?)\*/", text)
  if block_match:
    comment = block_match.group(1).strip()
    remainder = text[block_match.end():].lstrip()

    return comment, remainder

  lines = text.splitlines()
  comment_lines = []
  idx = 0
  while idx < len(lines):
    line = lines[idx]
    if not line.strip() and not comment_lines:
      idx += 1
      continue
    stripped = line.lstrip()
    if stripped.startswith("//") or stripped.startswith("#"):
      if stripped.startswith("#include") or stripped.startswith("#if"):
        break
      comment_lines.append(stripped)
      idx += 1
      continue
    break

  if comment_lines:
    comment = "\n".join(_strip_comment_prefix(l) for l in comment_lines).strip()
    remainder = "\n".join(lines[idx:]).lstrip()
    return comment, remainder

  return None, text


def _extract_leading_comment(text: str) -> str:
  comment, _ = _split_leading_comment(text)
  return comment or ""


def parse_freeform_response(response_text: str) -> dict:
  """Parse a free-text LLM response into discussion + code."""
  if not response_text:
    return {"discussion": "", "code": ""}

  text = response_text.strip()
  fence_re = re.compile(r"```(?:[a-zA-Z0-9_+-]*)[ \t]*\r?\n([\s\S]*?)```", re.MULTILINE)
  match = fence_re.search(text)
  if match:
    discussion = text[:match.start()].strip()
    code = match.group(1).strip("\r\n")
    if not discussion:
      discussion = _extract_leading_comment(code)
    return {"discussion": discussion, "code": code}

  discussion, code = _split_leading_comment(text)
  if discussion is None:
    discussion = ""
    code = text
  return {"discussion": discussion, "code": code.strip("\r\n")}


def normalize_code_result(result: Any, code_key: str) -> dict:
  """Return a legacy-shaped dict for graders from structured or freeform output."""
  if isinstance(result, dict):
    normalized = dict(result)
    if code_key not in normalized:
      for fallback_key in (
          "code",
          "cpp_code",
          "csharp_code",
          "rust_code",
          "python_code",
          "shader_code",
          "spirv_code",
          "spirv_hex",
      ):
        value = normalized.get(fallback_key)
        if isinstance(value, str) and value.strip():
          normalized[code_key] = value
          break
    if "reasoning" not in normalized:
      discussion = normalized.get("discussion") or normalized.get("reasoningAndDiscussion")
      if isinstance(discussion, str) and discussion:
        normalized["reasoning"] = discussion
    return normalized

  if isinstance(result, str):
    parsed = parse_freeform_response(result)
    code = parsed.get("code", "")
    discussion = parsed.get("discussion", "")
    if not code and not discussion:
      return {}
    normalized = {}
    if code:
      normalized[code_key] = code
    if discussion:
      normalized["reasoning"] = discussion
    return normalized

  return {}


class GradeCache:
  """Disk-based cache for gradeAnswer and resultToNiceReport results.

  Cache key is derived from a hash of the code being graded plus the
  test-case parameters (graph seed, size, etc.).
  """

  def __init__(self, test_name: str):
    self.cache_dir = GRADE_CACHE_DIR / test_name
    self.cache_dir.mkdir(parents=True, exist_ok=True)

  @staticmethod
  def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
      h.update(p.encode('utf-8'))
    return h.hexdigest()[:32]

  def _path(self, key_hash: str, kind: str) -> Path:
    kind = re.sub(r"[^A-Za-z0-9_.-]", "_", kind)
    return self.cache_dir / f"{key_hash}_{kind}.json"

  def _lock_path(self, key_hash: str, kind: str) -> Path:
    kind = re.sub(r"[^A-Za-z0-9_.-]", "_", kind)
    return self.cache_dir / f"{key_hash}_{kind}.lock"

  @staticmethod
  def _read_json_path(path: Path):
    if path.exists():
      try:
        return json.loads(path.read_text(encoding='utf-8'))
      except Exception:
        pass
    return None

  @staticmethod
  def _write_json_atomic(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
      with tempfile.NamedTemporaryFile(mode='w',
                                       encoding='utf-8',
                                       dir=path.parent,
                                       prefix=path.name + ".",
                                       suffix=".tmp",
                                       delete=False) as f:
        tmp_path = Path(f.name)
        json.dump(data, f)
        f.flush()
        try:
          os.fsync(f.fileno())
        except Exception:
          pass
      os.replace(tmp_path, path)
    finally:
      if tmp_path is not None:
        try:
          tmp_path.unlink(missing_ok=True)
        except Exception:
          pass

  @staticmethod
  def _read_lock_info(lock_path: Path) -> dict:
    try:
      text = lock_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
      return {"pid": None, "time": None, "raw": "", "read_error": repr(e)}

    pid = None
    created_at = None
    pid_match = re.search(r"\bpid=(\d+)\b", text)
    time_match = re.search(r"\btime=([0-9.]+)\b", text)
    if pid_match:
      try:
        pid = int(pid_match.group(1))
      except ValueError:
        pid = None
    if time_match:
      try:
        created_at = float(time_match.group(1))
      except ValueError:
        created_at = None
    return {"pid": pid, "time": created_at, "raw": text.strip()}

  @staticmethod
  def _pid_is_running(pid: Optional[int]) -> Optional[bool]:
    if pid is None or pid <= 0:
      return None
    if pid == os.getpid():
      return True

    if platform.system() == 'Windows':
      try:
        import ctypes
        import ctypes.wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD
        ]
        kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
        kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
        process_query_limited_information = 0x1000
        handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if handle:
          kernel32.CloseHandle(handle)
          return True
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: no such PID.
          return False
        if error == 5:  # ERROR_ACCESS_DENIED: PID exists, but we cannot inspect it.
          return True
      except Exception:
        return None
      return None

    try:
      os.kill(pid, 0)
      return True
    except ProcessLookupError:
      return False
    except PermissionError:
      return True
    except OSError:
      return None

  class _LockToken:

    def __init__(self, lock_path: Path, heartbeat_interval: float):
      self.lock_path = lock_path
      self._stop_event = threading.Event()
      self._thread = threading.Thread(target=self._heartbeat,
                                      args=(heartbeat_interval,),
                                      daemon=True)
      self._thread.start()

    def _heartbeat(self, heartbeat_interval: float):
      while not self._stop_event.wait(heartbeat_interval):
        try:
          os.utime(self.lock_path, None)
        except FileNotFoundError:
          return
        except Exception:
          pass

    def close(self):
      self._stop_event.set()
      self._thread.join(timeout=1.0)

  @staticmethod
  def _acquire_lock(lock_path: Path,
                    poll_interval: float = 0.1,
                    stale_seconds: float = 12 * 60 * 60,
                    malformed_stale_seconds: float = 60,
                    reclaim_grace_seconds: float = 30):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    reclaimable_since = None
    last_reclaim_error = None

    while True:
      try:
        stat = lock_path.stat()
      except FileNotFoundError:
        fd = None
        try:
          fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
          os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
          try:
            os.fsync(fd)
          except Exception:
            pass
          heartbeat_interval = max(1.0, min(60.0, stale_seconds / 4.0))
          return GradeCache._LockToken(lock_path, heartbeat_interval)
        except FileExistsError:
          pass
        except Exception:
          if fd is not None:
            try:
              lock_path.unlink(missing_ok=True)
            except Exception:
              pass
          raise
        finally:
          if fd is not None:
            try:
              os.close(fd)
            except Exception:
              pass
      else:
        now = time.time()
        age = max(0.0, now - stat.st_mtime)
        info = GradeCache._read_lock_info(lock_path)
        owner_running = GradeCache._pid_is_running(info.get("pid"))
        malformed = info.get("pid") is None
        should_reclaim = (
            owner_running is False
            or age > stale_seconds
            or (malformed and age > malformed_stale_seconds)
        )

        if should_reclaim:
          if reclaimable_since is None:
            reclaimable_since = now
          try:
            lock_path.unlink()
            reclaimable_since = None
            last_reclaim_error = None
            continue
          except FileNotFoundError:
            reclaimable_since = None
            last_reclaim_error = None
            continue
          except Exception as e:
            last_reclaim_error = e
            if now - reclaimable_since >= reclaim_grace_seconds:
              owner_desc = f"pid={info.get('pid')}"
              if owner_running is False:
                owner_desc += " (not running)"
              elif owner_running is True:
                owner_desc += " (running)"
              else:
                owner_desc += " (unknown)"
              raise RuntimeError(
                  f"Unable to reclaim cache lock {lock_path} after "
                  f"{now - reclaimable_since:.1f}s; {owner_desc}, "
                  f"age={age:.1f}s, raw={info.get('raw')!r}, "
                  f"delete_error={last_reclaim_error!r}") from e
        else:
          reclaimable_since = None
          last_reclaim_error = None

      time.sleep(poll_interval)

  @staticmethod
  def _release_lock(lock_path: Path, lock_token):
    if hasattr(lock_token, "close"):
      try:
        lock_token.close()
      except Exception:
        pass
    else:
      try:
        os.close(lock_token)
      except Exception:
        pass
    try:
      lock_path.unlink(missing_ok=True)
    except Exception:
      pass

  def get_json(self, kind: str, *key_parts: str):
    h = self._hash_key(*key_parts)
    return self._read_json_path(self._path(h, kind))

  def put_json(self, kind: str, data, *key_parts: str):
    h = self._hash_key(*key_parts)
    self._write_json_atomic(self._path(h, kind), data)

  def get_or_compute_json(self, kind: str, compute: Callable[[], Any], *key_parts: str):
    h = self._hash_key(*key_parts)
    path = self._path(h, kind)
    cached = self._read_json_path(path)
    if cached is not None:
      return cached

    lock_path = self._lock_path(h, kind)
    lock_token = self._acquire_lock(lock_path)
    try:
      cached = self._read_json_path(path)
      if cached is not None:
        return cached
      data = compute()
      self._write_json_atomic(path, data)
      return data
    finally:
      self._release_lock(lock_path, lock_token)

  def get_grade(self, *key_parts: str):
    """Return cached (score, details) or None."""
    h = self._hash_key(*key_parts)
    data = self._read_json_path(self._path(h, "grade"))
    if data is not None:
      try:
        return (data["score"], data["details"])
      except Exception:
        pass
    return None

  def put_grade(self, result, *key_parts: str):
    """Cache a (score, details) tuple."""
    h = self._hash_key(*key_parts)
    try:
      self._write_json_atomic(self._path(h, "grade"), {"score": result[0], "details": result[1]})
    except Exception:
      pass

  def get_report(self, *key_parts: str):
    """Return cached HTML string or None."""
    h = self._hash_key(*key_parts)
    p = self._path(h, "report")
    if p.exists():
      try:
        return p.read_text(encoding='utf-8')
      except Exception:
        pass
    return None

  def put_report(self, html: str, *key_parts: str):
    """Cache an HTML report string."""
    h = self._hash_key(*key_parts)
    p = self._path(h, "report")
    try:
      p.write_text(html, encoding='utf-8')
    except Exception:
      pass


class BaselineCache(GradeCache):
  """Disk-based cache for expensive baseline/reference computations."""

  def __init__(self, test_name: str):
    self.cache_dir = BASELINE_CACHE_DIR / test_name
    self.cache_dir.mkdir(parents=True, exist_ok=True)


class StreamingInputFile:
  """
  Manages streaming input data that may be too large to fit in memory.
  
  Supports:
  - Caching generated data to temp files
  - Lazy generation via generators/callables
  - Automatic cache invalidation via content hash
  - File handle access for subprocess stdin
  """

  def __init__(self,
               cache_key: str,
               generator: Callable[[], Generator[str, None, None]],
               cache_subdir: str = "default"):
    """
    Args:
      cache_key: Unique key for caching (should include all parameters that affect output)
      generator: Callable that returns a generator yielding string chunks
      cache_subdir: Subdirectory within cache for organization
    """
    self.cache_key = cache_key
    self.generator = generator
    self.cache_subdir = cache_subdir
    self._file_path: Optional[Path] = None
    self._is_generated = False

  @property
  def cache_dir(self) -> Path:
    return STREAMING_CACHE_DIR / self.cache_subdir

  @property
  def cache_path(self) -> Path:
    key_hash = hashlib.sha256(self.cache_key.encode('utf-8')).hexdigest()[:24]
    return self.cache_dir / f"{key_hash}.txt"

  def _ensure_cache_dir(self):
    self.cache_dir.mkdir(parents=True, exist_ok=True)

  def is_cached(self) -> bool:
    return self.cache_path.exists()

  def generate(self, force: bool = False) -> Path:
    """
    Generate the data file if not cached.
    
    Args:
      force: If True, regenerate even if cached
      
    Returns:
      Path to the data file
    """
    if not force and self.is_cached():
      self._file_path = self.cache_path
      self._is_generated = True
      return self._file_path

    self._ensure_cache_dir()
    tmp_path = self.cache_path.with_suffix('.tmp')

    try:
      with open(tmp_path, 'w', encoding='utf-8', newline='\n') as f:
        for chunk in self.generator():
          f.write(chunk)

      # Atomic rename
      os.replace(tmp_path, self.cache_path)
      self._file_path = self.cache_path
      self._is_generated = True
      return self._file_path

    except Exception:
      # Clean up partial file
      try:
        tmp_path.unlink(missing_ok=True)
      except Exception:
        pass
      raise

  def get_file_path(self) -> Path:
    """Get path to data file, generating if needed."""
    if not self._is_generated:
      self.generate()
    return self._file_path

  def open_for_read(self):
    """Open the data file for reading (for subprocess stdin)."""
    return open(self.get_file_path(), 'r', encoding='utf-8')

  def get_size_bytes(self) -> int:
    """Get file size in bytes."""
    return self.get_file_path().stat().st_size

  def delete_cache(self):
    """Delete the cached file."""
    try:
      self.cache_path.unlink(missing_ok=True)
    except Exception:
      pass
    self._is_generated = False
    self._file_path = None


def streaming_graph_generator(num_nodes: int,
                              num_edges: int,
                              edge_generator: Callable[[], Iterator[tuple]],
                              chunk_size: int = 10000) -> Generator[str, None, None]:
  """
  Generator that yields graph data in chunks suitable for streaming.
  
  Args:
    num_nodes: Number of nodes
    num_edges: Number of edges  
    edge_generator: Callable returning iterator of (u, v) edge tuples
    chunk_size: Number of edges per chunk
    
  Yields:
    String chunks of graph data
  """
  # Header line
  yield f"{num_nodes} {num_edges}\n"

  # Edge lines in chunks
  buffer = []
  for u, v in edge_generator():
    buffer.append(f"{u} {v}\n")
    if len(buffer) >= chunk_size:
      yield ''.join(buffer)
      buffer.clear()

  # Remaining edges
  if buffer:
    yield ''.join(buffer)


def create_graph_input_file(num_nodes: int,
                            num_edges: int,
                            cut_size: int,
                            seed: int,
                            graph_generator_func: Callable,
                            cache_subdir: str = "graphs") -> StreamingInputFile:
  """
  Create a StreamingInputFile for graph data.
  
  Args:
    num_nodes: Number of nodes
    num_edges: Number of edges
    cut_size: Size of the known cut
    seed: Random seed
    graph_generator_func: Function(num_nodes, num_edges, cut_size, seed) -> (edges, actual_cut)
    cache_subdir: Cache subdirectory
    
  Returns:
    StreamingInputFile that can generate/cache the graph data
  """
  cache_key = f"graph|n={num_nodes}|m={num_edges}|cut={cut_size}|seed={seed}"

  def generator():
    edges, _ = graph_generator_func(num_nodes, num_edges, cut_size, seed)

    yield f"{num_nodes} {len(edges) if hasattr(edges, '__len__') else num_edges}\n"

    # Handle both numpy arrays and lists of tuples
    if np is not None and hasattr(edges, 'shape'):
      # NumPy array - iterate efficiently
      for i in range(edges.shape[0]):
        yield f"{int(edges[i, 0])} {int(edges[i, 1])}\n"
    else:
      # List of tuples
      for u, v in edges:
        yield f"{u} {v}\n"

  return StreamingInputFile(cache_key, generator, cache_subdir)


def clean_streaming_cache(subdir: str = None):
  """
  Clean streaming cache.
  
  Args:
    subdir: Specific subdirectory to clean, or None for all
  """
  if subdir:
    target = STREAMING_CACHE_DIR / subdir
  else:
    target = STREAMING_CACHE_DIR

  if target.exists():
    shutil.rmtree(target)


def create_isolated_environment():
  """Create a completely isolated environment for subprocess execution."""
  clean_env = {
    'PATH': os.environ.get('PATH', ''),
    'SYSTEMROOT': os.environ.get('SYSTEMROOT', ''),  # Windows compatibility
    'PYTHONPATH': '',
    'PYDEVD_DISABLE_FILE_VALIDATION': '1',
  }

  # Remove ALL debugger-related variables from PATH and other critical vars
  critical_vars = ['PATH', 'PYTHONPATH', 'SystemDrive', 'SystemRoot']
  for var in critical_vars:
    if var in os.environ:
      clean_env[var] = os.environ[var]

  return clean_env


def build_python_command(temp_file):
  """Build the Python command with debugger bypass flags."""
  if platform.system() == 'Windows':
    # On Windows, try to use cmd /c to bypass debugger
    # Use subprocess.Popen for better control
    python_exe = sys.executable

    # Build command as a list to avoid quoting issues
    cmd_args = ['cmd', '/c', python_exe, '-Xfrozen_modules=off', temp_file]
    return cmd_args
  else:
    # On Unix systems, use direct execution
    python_cmd = [sys.executable, '-Xfrozen_modules=off', '-S']
    return python_cmd + [temp_file]


def execute_solver_code(code_content, temp_file_content, timeout=30):
  """
    Execute LLM solver code with debugger isolation.
    
    Args:
        code_content: The LLM-generated solver code (as string)
        temp_file_content: The complete content to write to temp file (as string)
        timeout: Timeout in seconds
        
    Returns:
        tuple: (result, error, execution_time)
  """

  try:
    compileTest = compile(code_content, '<LLM_GeneratedCode>', 'exec')
  except SyntaxError as e:
    return None, f"Syntax error in solver code: {e}", 0.0

  # Create temporary file with the solver code
  with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
    f.write(temp_file_content)
    temp_path = f.name

  process = None
  try:
    # Create isolated environment
    clean_env = create_isolated_environment()

    # Pre-generate .pyc outside the timed section to reduce first-run compile overhead
    try:
      if platform.system() == 'Windows':
        compile_cmd = [
          'cmd', '/c', sys.executable, '-Xfrozen_modules=off', '-m', 'py_compile', temp_path
        ]
      else:
        compile_cmd = [sys.executable, '-Xfrozen_modules=off', '-S', '-m', 'py_compile', temp_path]

      subprocess.run(compile_cmd,
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.PIPE,
                     text=True,
                     env=clean_env,
                     cwd=tempfile.gettempdir(),
                     timeout=min(10, max(1, timeout // 3)))
    except Exception:
      pass

    # Build command with debugger bypass
    cmd_args = build_python_command(temp_path)

    start_time = time.time()

    # Start the process
    if platform.system() == 'Windows':
      # On Windows, we need to create a new process group to be able to kill it
      process = subprocess.Popen(cmd_args,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 text=True,
                                 env=clean_env,
                                 cwd=tempfile.gettempdir(),
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
      # On Unix, we can use a process group
      process = subprocess.Popen(cmd_args,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 text=True,
                                 env=clean_env,
                                 cwd=tempfile.gettempdir(),
                                 preexec_fn=os.setsid)

    try:
      # Wait for the process with timeout
      stdout, stderr = process.communicate(timeout=timeout)
      execution_time = time.time() - start_time
    except subprocess.TimeoutExpired:
      execution_time = time.time() - start_time
      print(f"Timeout expired. {timeout} seconds")

      # Kill the process and all its children
      if platform.system() == 'Windows':
        # On Windows, kill the entire process tree
        subprocess.call(
          ['taskkill', '/F', '/T', '/PID', str(process.pid)],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL)
      else:
        # On Unix, kill the process group
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)

      process.wait()  # Clean up the zombie process
      return None, f"Timeout: solver exceeded {timeout} seconds", execution_time

    if execution_time > 1:
      print(f"Solver took : {execution_time:.2f} seconds")

    if process.returncode != 0:
      error = stderr.strip() if stderr else "Unknown error"
      print(f"Solver crashed: {error[:500]}")
      return None, f"Solver crashed: {error[:500]}", execution_time

    try:
      output = json.loads(stdout.strip())
      if isinstance(output, dict) and 'error' in output:
        error_msg = output['error']
        if 'traceback' in output:
          print(f"Solver error: {error_msg}")
          print(f"Exception type: {output.get('exception_type', 'Unknown')}")
          print(f"Full traceback:\n{output['traceback']}")
          print(f"Command: {' '.join(cmd_args)}")
        else:
          print(f"Solver error: {error_msg}")
          print(f"Command: {' '.join(cmd_args)}")
        return None, f"Solver error: {error_msg}", execution_time
      return output, None, execution_time
    except json.JSONDecodeError as e:
      print(f"Invalid output: {stdout[:500]} - Error: {e}")
      return None, f"Invalid output: {stdout[:200]} - Error: {e}", execution_time

  except Exception as e:
    print(f"Execution error: {str(e)}")
    return None, f"Execution error: {str(e)}", 0
  finally:
    # Ensure the process is killed if it's still running
    if process and process.poll() is None:
      try:
        if platform.system() == 'Windows':
          subprocess.call(
            ['taskkill', '/F', '/T', '/PID', str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        else:
          os.killpg(os.getpgid(process.pid), signal.SIGKILL)
      except:
        pass

    # Clean up the temp file
    try:
      os.unlink(temp_path)
    except:
      pass


def execute_solver_with_data(code_content, data_dict, solver_function_name, timeout=30):
  """
    Execute LLM solver code with data parameters.
    
    Args:
        code_content: The LLM-generated solver code (as string)
        data_dict: Dictionary of data to include in the temp file
        solver_function_name: Name of the solver function to call
        timeout: Timeout in seconds
        
    Returns:
        tuple: (result, error, execution_time)
    """

  # Remove any existing __main__ block from LLM code
  lines = code_content.split('\n')
  cleaned_lines = []
  in_main_block = False
  main_block_indent = 0

  for line in lines:
    stripped = line.strip()

    # Detect start of __main__ block
    if (stripped.startswith('if __name__')
        and ('__main__' in stripped or '"__main__"' in stripped)):
      in_main_block = True
      main_block_indent = len(line) - len(line.lstrip())
      continue

    # If we're in a __main__ block, check if we've exited it
    if in_main_block:
      # Only check for exit if we have a non-empty line
      if stripped:
        current_indent = len(line) - len(line.lstrip())
        # Exit when we hit a line with same or less indentation than the if statement
        if current_indent <= main_block_indent:
          in_main_block = False
          # Add this line since it's outside the block
          cleaned_lines.append(line)
        else:
          # Still inside the block, skip this line
          continue
      else:
        # Empty line inside the block, skip it
        continue

    # Add line if not in __main__ block
    cleaned_lines.append(line)

  cleaned_code = '\n'.join(cleaned_lines)

  # Build temp file content
  temp_content = "import sys\n"
  temp_content += "import json\n"
  temp_content += "\n"
  temp_content += "def _json_safe(o):\n"
  temp_content += "    try:\n"
  temp_content += "        import numpy as _np\n"
  temp_content += "        if isinstance(o, _np.generic):\n"
  temp_content += "            return o.item()\n"
  temp_content += "        if isinstance(o, _np.ndarray):\n"
  temp_content += "            return o.tolist()\n"
  temp_content += "    except Exception:\n"
  temp_content += "        pass\n"
  temp_content += "    if isinstance(o, dict):\n"
  temp_content += "        return {str(k): _json_safe(v) for k, v in o.items()}\n"
  temp_content += "    if isinstance(o, (list, tuple)):\n"
  temp_content += "        return [_json_safe(v) for v in o]\n"
  temp_content += "    if isinstance(o, set):\n"
  temp_content += "        return [_json_safe(v) for v in o]\n"
  temp_content += "    if isinstance(o, (str, int, float, bool)) or o is None:\n"
  temp_content += "        return o\n"
  temp_content += "    if hasattr(o, 'tolist'):\n"
  temp_content += "        try:\n"
  temp_content += "            return o.tolist()\n"
  temp_content += "        except Exception:\n"
  temp_content += "            pass\n"
  temp_content += "    if hasattr(o, 'item'):\n"
  temp_content += "        try:\n"
  temp_content += "            return o.item()\n"
  temp_content += "        except Exception:\n"
  temp_content += "            pass\n"
  temp_content += "    return str(o)\n"

  # Add data variables
  for key, value in data_dict.items():
    temp_content += f"\n{key} = {value!r}\n"

  assert "\nif __name__ == '__main__':" not in cleaned_code
  assert "\n" + 'if __name__ == "__main__":' not in cleaned_code

  temp_content += "\n# LLM-generated solver code\n"
  temp_content += cleaned_code
  temp_content += f"\n\n# Execute and output result\n"
  temp_content += "if __name__ == '__main__':\n"
  temp_content += "    try:\n"
  temp_content += f"        result = {solver_function_name}({', '.join(data_dict.keys())})\n"
  temp_content += "        print(json.dumps(_json_safe(result)))\n"
  temp_content += "    except Exception as e:\n"
  temp_content += "        import traceback\n"
  temp_content += "        error_info = {\n"
  temp_content += "            'error': str(e),\n"
  temp_content += "            'traceback': traceback.format_exc(),\n"
  temp_content += "            'exception_type': type(e).__name__\n"
  temp_content += "        }\n"
  temp_content += "        print(json.dumps(_json_safe(error_info)))\n"

  return execute_solver_code(cleaned_code, temp_content, timeout)
