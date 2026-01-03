"""
Native Code Compiler Helpers

Provides helper classes for compiling C++, Rust, and C# code to native executables.
Supports Windows (Visual C++, MSVC, MinGW) and Linux (GCC, Clang) environments.
Includes caching to avoid recompilation when source hasn't changed.
"""

import hashlib
import ctypes
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import signal
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable

# Base directory for compiled outputs
COMPILE_CACHE_DIR = Path(__file__).parent / "compile_cache"

DESCRIPTION_CACHE_DIR = Path(tempfile.gettempdir()) / "codingbenchmark_native_compiler_desc_cache"


def get_file_hash(content: str) -> str:
  """Generate hash of source code for caching."""
  return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def ensure_dir(path: Path) -> Path:
  """Ensure directory exists."""
  path.mkdir(parents=True, exist_ok=True)
  return path


def _safe_stat_fingerprint(path: str) -> str:
  try:
    st = os.stat(path)
    return f"{st.st_mtime_ns}:{st.st_size}"
  except Exception:
    return "unknown"


def _read_or_compute_description(cache_kind: str, cache_key: str, compute: Callable[[],
                                                                                    str]) -> str:
  cache_dir = ensure_dir(DESCRIPTION_CACHE_DIR / cache_kind)
  cache_path = cache_dir / f"{get_file_hash(cache_key)}.txt"
  try:
    if cache_path.exists():
      return cache_path.read_text(encoding='utf-8', errors='replace')
  except Exception:
    pass

  value = compute()
  try:
    tmp_path = cache_path.with_suffix('.tmp')
    tmp_path.write_text(value, encoding='utf-8')
    os.replace(tmp_path, cache_path)
  except Exception:
    pass
  return value


class CompilationError(Exception):
  """Raised when compilation fails."""
  pass


class ExecutionError(Exception):
  """Raised when execution fails."""
  pass


def describe_this_pc():

  cache_key = "|".join([
    platform.system(),
    platform.release(),
    platform.version(),
    platform.machine(),
    platform.processor() or "unknown",
    str(os.cpu_count() or 0),
  ])

  def _compute() -> str:

    def get_memory_gb() -> str:
      try:
        if platform.system() == 'Windows':

          class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
              ('dwLength', ctypes.c_uint32),
              ('dwMemoryLoad', ctypes.c_uint32),
              ('ullTotalPhys', ctypes.c_uint64),
              ('ullAvailPhys', ctypes.c_uint64),
              ('ullTotalPageFile', ctypes.c_uint64),
              ('ullAvailPageFile', ctypes.c_uint64),
              ('ullTotalVirtual', ctypes.c_uint64),
              ('ullAvailVirtual', ctypes.c_uint64),
              ('ullAvailExtendedVirtual', ctypes.c_uint64),
            ]

          status = MEMORYSTATUSEX()
          status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
          if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return f"{status.ullTotalPhys / (1024**3):.2f}"
        else:
          if hasattr(os, 'sysconf'):
            page = os.sysconf('SC_PAGE_SIZE')
            pages = os.sysconf('SC_PHYS_PAGES')
            return f"{(page * pages) / (1024**3):.2f}"
      except Exception:
        pass
      return "unknown"

    def get_simd_features() -> Dict[str, Optional[bool]]:
      system = platform.system()
      feats: Dict[str, Optional[bool]] = {}

      if system == 'Windows':
        try:
          kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
          is_present = kernel32.IsProcessorFeaturePresent
          is_present.argtypes = [ctypes.c_uint32]
          is_present.restype = ctypes.c_int

          PF_XMMI_INSTRUCTIONS_AVAILABLE = 6
          PF_XMMI64_INSTRUCTIONS_AVAILABLE = 10
          PF_SSE3_INSTRUCTIONS_AVAILABLE = 13
          PF_SSSE3_INSTRUCTIONS_AVAILABLE = 36
          PF_SSE4_1_INSTRUCTIONS_AVAILABLE = 37
          PF_SSE4_2_INSTRUCTIONS_AVAILABLE = 38
          PF_AVX_INSTRUCTIONS_AVAILABLE = 39
          PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
          PF_AVX512F_INSTRUCTIONS_AVAILABLE = 41

          checks = {
            'SSE': PF_XMMI_INSTRUCTIONS_AVAILABLE,
            'SSE2': PF_XMMI64_INSTRUCTIONS_AVAILABLE,
            'SSE3': PF_SSE3_INSTRUCTIONS_AVAILABLE,
            'SSSE3': PF_SSSE3_INSTRUCTIONS_AVAILABLE,
            'SSE4.1': PF_SSE4_1_INSTRUCTIONS_AVAILABLE,
            'SSE4.2': PF_SSE4_2_INSTRUCTIONS_AVAILABLE,
            'AVX': PF_AVX_INSTRUCTIONS_AVAILABLE,
            'AVX2': PF_AVX2_INSTRUCTIONS_AVAILABLE,
            'AVX-512F': PF_AVX512F_INSTRUCTIONS_AVAILABLE,
          }

          for name, pfid in checks.items():
            feats[name] = bool(is_present(pfid))

          return feats
        except Exception:
          return feats

      if system == 'Linux':
        try:
          flags = set()
          with open('/proc/cpuinfo', 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
              if line.lower().startswith('flags'):
                _, _, rhs = line.partition(':')
                flags.update(rhs.strip().split())
                break

          feats['SSE'] = 'sse' in flags
          feats['SSE2'] = 'sse2' in flags
          feats['SSE3'] = 'sse3' in flags
          feats['SSSE3'] = 'ssse3' in flags
          feats['SSE4.1'] = 'sse4_1' in flags
          feats['SSE4.2'] = 'sse4_2' in flags
          feats['AVX'] = 'avx' in flags
          feats['AVX2'] = 'avx2' in flags
          feats['AVX-512F'] = 'avx512f' in flags
        except Exception:
          pass
        return feats

      return feats

    mem_gb = get_memory_gb()
    simd = get_simd_features()

    simd_known = [k for k, v in simd.items() if v is True]
    simd_known.sort()

    simd_lines = []
    if simd:
      for k in sorted(simd.keys()):
        v = simd[k]
        if v is True:
          vv = 'yes'
        elif v is False:
          vv = 'no'
        else:
          vv = 'unknown'
        simd_lines.append(f"{k}={vv}")

    simd_summary = ', '.join(simd_known) if simd_known else 'unknown'
    simd_detail = ("\nSIMD detail: " + ', '.join(simd_lines)) if simd_lines else ""

    return f"""
Platform: {platform.system()} {platform.release()} (build {platform.version()})
Architecture: {platform.machine()}
CPU: {platform.processor() or 'unknown'}
Core count: {os.cpu_count()}
Memory: {mem_gb} GB
SIMD: {simd_summary}{simd_detail}
    """.strip()

  return _read_or_compute_description('this_pc', cache_key, _compute)


class NativeCompiler(ABC):
  """Abstract base class for native code compilers."""

  def __init__(self, engine_name: str, language: str):
    """
        Initialize compiler.
        
        Args:
            engine_name: Name of the AI engine (for cache separation)
            language: Programming language (cpp, rust, csharp)
        """
    self.engine_name = self._sanitize_name(engine_name)
    self.language = language
    self.is_windows = platform.system() == "Windows"
    self.is_linux = platform.system() == "Linux"
    self.cache_dir = ensure_dir(COMPILE_CACHE_DIR / self.engine_name / language)

  def _sanitize_name(self, name: str) -> str:
    """Sanitize engine name for use in file paths."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

  def _get_exe_extension(self) -> str:
    """Get executable extension for current platform."""
    return ".exe" if self.is_windows else ""

  def _get_cached_exe_path(self, source_hash: str) -> Path:
    """Get path to cached executable."""
    return self.cache_dir / f"{source_hash}{self._get_exe_extension()}"

  def _is_cached(self, source_hash: str) -> bool:
    """Check if compiled executable exists in cache."""
    exe_path = self._get_cached_exe_path(source_hash)
    return exe_path.exists()

  @abstractmethod
  def find_compiler(self) -> Optional[str]:
    """Find compiler executable. Returns path or None if not found."""
    pass

  @abstractmethod
  def compile(self, source_code: str, extra_flags: List[str] = None) -> Path:
    """
        Compile source code to executable.
        
        Args:
            source_code: Source code string
            extra_flags: Additional compiler flags
            
        Returns:
            Path to compiled executable
            
        Raises:
            CompilationError: If compilation fails
        """
    pass

  def execute(self,
              exe_path: Path,
              stdin_data: str = "",
              timeout: float = 300,
              env: Dict[str, str] = None,
              stdin_file: Optional[Path] = None) -> Tuple[str, str, float, int]:
    """
        Execute compiled program.
        
        Args:
            exe_path: Path to executable
            stdin_data: Data to send to stdin (ignored if stdin_file provided)
            timeout: Timeout in seconds
            env: Environment variables
            stdin_file: Path to file to use as stdin (for streaming large inputs)
            
        Returns:
            Tuple of (stdout, stderr, execution_time, return_code)
            
        Raises:
            ExecutionError: If execution fails or times out
        """
    start_time = time.time()
    process = None
    stdin_handle = None

    try:
      if stdin_file is not None:
        # Use file-based stdin for streaming large inputs
        stdin_handle = open(stdin_file, 'rb')
        process = subprocess.Popen(
          [str(exe_path)],
          stdin=stdin_handle,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          env=env,
          creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if self.is_windows else 0)

        stdout, stderr = process.communicate(timeout=timeout)
      else:
        # Use string-based stdin (original behavior)
        process = subprocess.Popen(
          [str(exe_path)],
          stdin=subprocess.PIPE,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          env=env,
          creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if self.is_windows else 0)

        stdout, stderr = process.communicate(input=stdin_data.encode('utf-8'), timeout=timeout)

      execution_time = time.time() - start_time
      return (stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace'),
              execution_time, process.returncode)

    except subprocess.TimeoutExpired:
      execution_time = time.time() - start_time
      if process:
        self._kill_process(process)
      raise ExecutionError(f"Execution timed out after {timeout} seconds")

    except Exception as e:
      execution_time = time.time() - start_time
      if process:
        self._kill_process(process)
      raise ExecutionError(f"Execution error: {str(e)}")

    finally:
      if stdin_handle is not None:
        try:
          stdin_handle.close()
        except Exception:
          pass

  def _kill_process(self, process: subprocess.Popen):
    """Kill process and all its children."""
    try:
      if self.is_windows:
        # Use taskkill to kill process tree on Windows
        subprocess.run(
          ['taskkill', '/F', '/T', '/PID', str(process.pid)], capture_output=True, timeout=5)
      else:
        # Use process group kill on Linux
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


class CppCompiler(NativeCompiler):
  """C++ compiler supporting MSVC, GCC, Clang, and MinGW."""

  def __init__(self, engine_name: str):
    super().__init__(engine_name, "cpp")
    self._compiler_path = None
    self._compiler_type = None  # 'msvc', 'gcc', 'clang', 'mingw'
    self._detected_std = None
    self._detected_std_flag = None

  def _detect_cpp_standard_flag(self) -> str:
    if self._detected_std_flag:
      return self._detected_std_flag

    compiler = self.find_compiler()
    if not compiler:
      self._detected_std = "c++17"
      self._detected_std_flag = None
      return None

    if self._compiler_type == 'msvc':
      env = self._get_msvc_env()
      try:
        res = subprocess.run([compiler, '/?'], capture_output=True, text=True, env=env, timeout=10)
        out = (res.stdout or "") + "\n" + (res.stderr or "")
      except Exception:
        out = ""

      for flag in ("c++26", "c++23", "c++20", "c++17", "c++14"):
        if flag in out:
          self._detected_std = flag
          self._detected_std_flag = f"/std:{flag}"
          return self._detected_std_flag

      self._detected_std = "c++17"
      self._detected_std_flag = "/std:c++17"
      return self._detected_std_flag

    candidates = ("c++23", "c++20", "c++17", "c++14", "c++11")
    for std in candidates:
      src_path = None
      obj_path = None
      try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False,
                                         encoding='utf-8') as f:
          f.write('int main(){return 0;}')
          src_path = f.name
        obj_path = src_path + '.o'
        res = subprocess.run([compiler, f"-std={std}", '-c', src_path, '-o', obj_path],
                             capture_output=True,
                             text=True,
                             timeout=20)
        if res.returncode == 0:
          self._detected_std = std
          self._detected_std_flag = f"-std={std}"
          return self._detected_std_flag
      except Exception:
        pass
      finally:
        try:
          if src_path:
            os.unlink(src_path)
        except Exception:
          pass
        try:
          if obj_path:
            os.unlink(obj_path)
        except Exception:
          pass

    self._detected_std = "c++17"
    self._detected_std_flag = "-std=c++17"
    return self._detected_std_flag

  def find_compiler(self) -> Optional[str]:
    """Find C++ compiler in order of preference."""
    if self._compiler_path:
      return self._compiler_path

    if self.is_windows:
      # Try MSVC first (Visual Studio)
      msvc = self._find_msvc()
      if msvc:
        self._compiler_path = msvc
        self._compiler_type = 'msvc'
        return msvc

      # Try MinGW
      mingw = self._find_mingw()
      if mingw:
        self._compiler_path = mingw
        self._compiler_type = 'mingw'
        return mingw

      # Try Clang
      clang = self._find_clang()
      if clang:
        self._compiler_path = clang
        self._compiler_type = 'clang'
        return clang

    else:  # Linux
      # Try GCC first
      gcc = shutil.which('g++')
      if gcc:
        self._compiler_path = gcc
        self._compiler_type = 'gcc'
        return gcc

      # Try Clang
      clang = shutil.which('clang++')
      if clang:
        self._compiler_path = clang
        self._compiler_type = 'clang'
        return clang

    return None

  def _find_msvc(self) -> Optional[str]:
    """Find Visual Studio C++ compiler."""
    # Common Visual Studio installation paths
    vs_paths = [
      r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
      r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
      r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
      r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
      r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC",
      r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC",
    ]

    for vs_path in vs_paths:
      if os.path.exists(vs_path):
        # Find latest version
        try:
          versions = sorted(os.listdir(vs_path), reverse=True)
          for version in versions:
            cl_path = os.path.join(vs_path, version, "bin", "Hostx64", "x64", "cl.exe")
            if os.path.exists(cl_path):
              return cl_path
        except Exception:
          continue

    # Try vswhere
    try:
      vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
      if os.path.exists(vswhere):
        result = subprocess.run([vswhere, "-latest", "-property", "installationPath"],
                                capture_output=True,
                                text=True,
                                timeout=10)
        if result.returncode == 0:
          vs_install = result.stdout.strip()
          msvc_path = os.path.join(vs_install, "VC", "Tools", "MSVC")
          if os.path.exists(msvc_path):
            versions = sorted(os.listdir(msvc_path), reverse=True)
            for version in versions:
              cl_path = os.path.join(msvc_path, version, "bin", "Hostx64", "x64", "cl.exe")
              if os.path.exists(cl_path):
                return cl_path
    except Exception:
      pass

    return None

  def _find_mingw(self) -> Optional[str]:
    """Find MinGW G++ compiler."""
    mingw_paths = [
      r"C:\mingw64\bin\g++.exe",
      r"C:\msys64\mingw64\bin\g++.exe",
      r"C:\MinGW\bin\g++.exe",
      r"C:\TDM-GCC-64\bin\g++.exe",
    ]

    for path in mingw_paths:
      if os.path.exists(path):
        return path

    # Try PATH
    gpp = shutil.which('g++')
    if gpp:
      return gpp

    return None

  def _find_clang(self) -> Optional[str]:
    """Find Clang++ compiler."""
    clang_paths = [
      r"C:\Program Files\LLVM\bin\clang++.exe",
      r"C:\Program Files (x86)\LLVM\bin\clang++.exe",
    ]

    for path in clang_paths:
      if os.path.exists(path):
        return path

    return shutil.which('clang++')

  def _get_msvc_env(self) -> Dict[str, str]:
    """Get environment variables for MSVC compilation."""
    env = os.environ.copy()

    # Find vcvarsall.bat and run it to get environment
    try:
      vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
      if os.path.exists(vswhere):
        result = subprocess.run([vswhere, "-latest", "-property", "installationPath"],
                                capture_output=True,
                                text=True,
                                timeout=10)
        if result.returncode == 0:
          vs_install = result.stdout.strip()
          vcvars = os.path.join(vs_install, "VC", "Auxiliary", "Build", "vcvars64.bat")
          if os.path.exists(vcvars):
            # Run vcvars64.bat and capture environment
            cmd = f'cmd /c ""{vcvars}" && set"'
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True, timeout=30)
            if result.returncode == 0:
              for line in result.stdout.splitlines():
                if '=' in line:
                  key, _, value = line.partition('=')
                  env[key] = value
    except Exception:
      pass

    return env

  def compile(self, source_code: str, extra_flags: List[str] = None) -> Path:
    """Compile C++ source code."""
    source_hash = get_file_hash(source_code)

    # Check cache
    if self._is_cached(source_hash):
      return self._get_cached_exe_path(source_hash)

    compiler = self.find_compiler()
    if not compiler:
      raise CompilationError("No C++ compiler found")

    extra_flags = extra_flags or []
    exe_path = self._get_cached_exe_path(source_hash)

    # Create temp source file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
      f.write(source_code)
      src_path = f.name

    try:
      if self._compiler_type == 'msvc':
        # MSVC compilation
        env = self._get_msvc_env()
        obj_path = str(exe_path.with_suffix('.obj'))

        std_flag = self._detect_cpp_standard_flag()
        cmd = [
          compiler,
          '/nologo',
          '/EHsc',
          '/O2',
          std_flag if std_flag else '/std:c++17',
          src_path,
          f'/Fe:{exe_path}',
          f'/Fo:{obj_path}',
        ] + extra_flags

        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)

        # Clean up obj file
        try:
          os.unlink(obj_path)
        except Exception:
          pass

      else:
        # GCC/Clang/MinGW compilation
        std_flag = self._detect_cpp_standard_flag() or '-std=c++17'
        cmd = [
          compiler,
          '-O2',
          std_flag,
          '-o',
          str(exe_path),
          src_path,
        ] + extra_flags

        if self.is_linux:
          cmd.append('-pthread')

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

      if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown compilation error"
        raise CompilationError(f"Compilation failed:\n{error_msg[:2000]}")

      if not exe_path.exists():
        raise CompilationError("Compilation produced no output")

      return exe_path

    finally:
      try:
        os.unlink(src_path)
      except Exception:
        pass

  def describe(self):
    compiler = self.find_compiler()
    if not compiler:
      raise CompilationError("No C++ compiler found")

    cache_key = "|".join([
      platform.system(),
      platform.release(),
      platform.machine(),
      self._compiler_type or "unknown",
      compiler,
      _safe_stat_fingerprint(compiler),
    ])

    def _compute() -> str:
      std_flag = self._detect_cpp_standard_flag()
      std = self._detected_std or "unknown"

      try:
        if self._compiler_type == 'msvc':
          env = self._get_msvc_env()
          res = subprocess.run([compiler], capture_output=True, text=True, env=env, timeout=10)
          out = (res.stdout or "") + "\n" + (res.stderr or "")
          m = re.search(r"Compiler Version\s+([0-9.]+)", out)
          ver = m.group(1) if m else "unknown"
          return f"MSVC cl {ver} | std={std} ({std_flag})"

        res = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=10)
        first_line = (res.stdout or "").splitlines()[0].strip() if (
          res.stdout or "").splitlines() else "unknown"
        return f"{first_line} | std={std} ({std_flag})"
      except Exception:
        return f"{self._compiler_type} C++ compiler | std={std} ({std_flag})"

    return _read_or_compute_description('compiler_cpp', cache_key, _compute)


class RustCompiler(NativeCompiler):
  """Rust compiler using rustc or cargo."""

  def __init__(self, engine_name: str):
    super().__init__(engine_name, "rust")
    self._compiler_path = None
    self._detected_edition = None

  def _detect_rust_edition(self) -> str:
    if self._detected_edition:
      return self._detected_edition

    compiler = self.find_compiler()
    if not compiler:
      self._detected_edition = '2021'
      return self._detected_edition

    candidates = ('2024', '2021', '2018')
    for ed in candidates:
      src_path = None
      exe_path = None
      try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False,
                                         encoding='utf-8') as f:
          f.write('fn main() {}')
          src_path = f.name
        exe_path = src_path + ('.exe' if self.is_windows else '')
        res = subprocess.run([compiler, '-O', f'--edition={ed}', '-o', exe_path, src_path],
                             capture_output=True,
                             text=True,
                             timeout=30)
        if res.returncode == 0:
          self._detected_edition = ed
          return self._detected_edition
      except Exception:
        pass
      finally:
        try:
          if src_path:
            os.unlink(src_path)
        except Exception:
          pass
        try:
          if exe_path:
            os.unlink(exe_path)
        except Exception:
          pass

    self._detected_edition = '2021'
    return self._detected_edition

  def find_compiler(self) -> Optional[str]:
    """Find Rust compiler."""
    if self._compiler_path:
      return self._compiler_path

    # Try rustc
    rustc = shutil.which('rustc')
    if rustc:
      self._compiler_path = rustc
      return rustc

    # Common installation paths
    if self.is_windows:
      home = os.environ.get('USERPROFILE', '')
      rustc_path = os.path.join(home, '.cargo', 'bin', 'rustc.exe')
      if os.path.exists(rustc_path):
        self._compiler_path = rustc_path
        return rustc_path
    else:
      home = os.environ.get('HOME', '')
      rustc_path = os.path.join(home, '.cargo', 'bin', 'rustc')
      if os.path.exists(rustc_path):
        self._compiler_path = rustc_path
        return rustc_path

    return None

  def compile(self, source_code: str, extra_flags: List[str] = None) -> Path:
    """Compile Rust source code."""
    source_hash = get_file_hash(source_code)

    # Check cache
    if self._is_cached(source_hash):
      return self._get_cached_exe_path(source_hash)

    compiler = self.find_compiler()
    if not compiler:
      raise CompilationError("Rust compiler (rustc) not found")

    extra_flags = extra_flags or []
    exe_path = self._get_cached_exe_path(source_hash)

    # Create temp source file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
      f.write(source_code)
      src_path = f.name

    try:
      edition = self._detect_rust_edition()
      cmd = [
        compiler,
        '-O',
        f'--edition={edition}',
        '-o',
        str(exe_path),
        src_path,
      ] + extra_flags

      result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

      if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown compilation error"
        raise CompilationError(f"Compilation failed:\n{error_msg[:2000]}")

      if not exe_path.exists():
        raise CompilationError("Compilation produced no output")

      return exe_path

    finally:
      try:
        os.unlink(src_path)
      except Exception:
        pass

  def describe(self):
    """Return rust version and extensions."""
    compiler = self.find_compiler()
    if not compiler:
      raise CompilationError("Rust compiler (rustc) not found")

    cache_key = "|".join([
      platform.system(),
      platform.release(),
      platform.machine(),
      compiler,
      _safe_stat_fingerprint(compiler),
    ])

    def _compute() -> str:
      edition = self._detect_rust_edition()

      result = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=10)
      ver = result.stdout.strip() if result.stdout.strip() else "unknown"

      try:
        verbose = subprocess.run([compiler, "--version", "--verbose"],
                                 capture_output=True,
                                 text=True,
                                 timeout=10)
        extra = ""
        for line in (verbose.stdout or "").splitlines():
          if line.lower().startswith('host:'):
            extra = line.strip()
            break
        if extra:
          return f"{ver} | edition={edition} | {extra}"
      except Exception:
        pass

      return f"{ver} | edition={edition}"

    return _read_or_compute_description('compiler_rust', cache_key, _compute)


class CSharpCompiler(NativeCompiler):
  """C# compiler using .NET SDK (csc/dotnet) or Mono."""

  def __init__(self, engine_name: str):
    super().__init__(engine_name, "csharp")
    self._compiler_path = None
    self._compiler_type = None  # 'dotnet', 'csc', 'mono'

  def find_compiler(self) -> Optional[str]:
    """Find C# compiler."""
    if self._compiler_path:
      return self._compiler_path

    if self.is_windows:
      # Try .NET SDK first
      dotnet = self._find_dotnet()
      if dotnet:
        self._compiler_path = dotnet
        self._compiler_type = 'dotnet'
        return dotnet

      # Try csc.exe from .NET Framework
      csc = self._find_csc()
      if csc:
        self._compiler_path = csc
        self._compiler_type = 'csc'
        return csc

    else:  # Linux
      # Try .NET SDK
      dotnet = self._find_dotnet()
      if dotnet:
        self._compiler_path = dotnet
        self._compiler_type = 'dotnet'
        return dotnet

      # Try Mono
      mcs = shutil.which('mcs')
      if mcs:
        self._compiler_path = mcs
        self._compiler_type = 'mono'
        return mcs

    return None

  def _find_dotnet(self) -> Optional[str]:
    dotnet = shutil.which('dotnet')
    if dotnet:
      if self._dotnet_has_sdk(dotnet):
        return dotnet

    if self.is_windows:
      candidates = [
        r"C:\Program Files\dotnet\dotnet.exe",
        r"C:\Program Files (x86)\dotnet\dotnet.exe",
      ]
    else:
      candidates = [
        '/usr/bin/dotnet',
        '/usr/local/bin/dotnet',
      ]

    for p in candidates:
      if os.path.exists(p) and self._dotnet_has_sdk(p):
        return p

    return None

  def _dotnet_has_sdk(self, dotnet_path: str) -> bool:
    try:
      r = subprocess.run([dotnet_path, '--list-sdks'], capture_output=True, text=True, timeout=10)
      if r.returncode != 0:
        return False
      return bool(r.stdout.strip())
    except Exception:
      return False

  def _find_csc(self) -> Optional[str]:
    """Find csc.exe from .NET Framework or Roslyn."""
    # Try to find Roslyn csc
    try:
      vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
      if os.path.exists(vswhere):
        result = subprocess.run([vswhere, "-latest", "-property", "installationPath"],
                                capture_output=True,
                                text=True,
                                timeout=10)
        if result.returncode == 0:
          vs_install = result.stdout.strip()
          roslyn_csc = os.path.join(vs_install, "MSBuild", "Current", "Bin", "Roslyn", "csc.exe")
          if os.path.exists(roslyn_csc):
            return roslyn_csc
    except Exception:
      pass

    # .NET Framework paths (legacy fallback)
    framework_paths = [
      r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
      r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe",
    ]

    for path in framework_paths:
      if os.path.exists(path):
        return path

    return None

  def _get_exe_extension(self) -> str:
    """Get executable extension - .NET produces .dll that runs with dotnet."""
    if self._compiler_type == 'dotnet':
      return ".dll"
    return ".exe" if self.is_windows else ""

  def compile(self, source_code: str, extra_flags: List[str] = None) -> Path:
    """Compile C# source code."""
    source_hash = get_file_hash(source_code)

    # Check cache
    if self._is_cached(source_hash):
      return self._get_cached_exe_path(source_hash)

    compiler = self.find_compiler()
    if not compiler:
      raise CompilationError("No C# compiler found (dotnet, csc, or mcs)")

    extra_flags = extra_flags or []
    exe_path = self._get_cached_exe_path(source_hash)

    if self._compiler_type == 'dotnet':
      return self._compile_with_dotnet(source_code, source_hash, exe_path, extra_flags)
    elif self._compiler_type == 'csc':
      return self._compile_with_csc(source_code, exe_path, extra_flags)
    else:  # mono
      return self._compile_with_mono(source_code, exe_path, extra_flags)

  def _compile_with_dotnet(self, source_code: str, source_hash: str, exe_path: Path,
                           extra_flags: List[str]) -> Path:
    """Compile using .NET SDK."""
    # Create a temporary project
    project_dir = self.cache_dir / f"project_{source_hash}"
    ensure_dir(project_dir)

    try:
      # Write source file
      src_path = project_dir / "Program.cs"
      with open(src_path, 'w') as f:
        f.write(source_code)

      # Write minimal csproj
      csproj_content = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>"""
      csproj_path = project_dir / "Program.csproj"
      with open(csproj_path, 'w') as f:
        f.write(csproj_content)

      # Build
      result = subprocess.run(
        [self._compiler_path, 'build', '-c', 'Release', '-o',
         str(self.cache_dir)],
        cwd=str(project_dir),
        capture_output=True,
        text=True,
        timeout=120)

      if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown compilation error"
        raise CompilationError(f"Compilation failed:\n{error_msg[:2000]}")

      # The output is Program.dll
      dll_path = self.cache_dir / "Program.dll"
      if dll_path.exists():
        # Rename to our cached path
        target_path = exe_path
        shutil.copy2(dll_path, target_path)
        return target_path

      raise CompilationError("Compilation produced no output")

    finally:
      # Clean up project dir
      try:
        shutil.rmtree(project_dir)
      except Exception:
        pass

  def _compile_with_csc(self, source_code: str, exe_path: Path, extra_flags: List[str]) -> Path:
    """Compile using csc.exe."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cs', delete=False) as f:
      f.write(source_code)
      src_path = f.name

    try:
      cmd = [
        self._compiler_path,
        '/nologo',
        '/optimize+',
        f'/out:{exe_path}',
        src_path,
      ] + extra_flags

      result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

      if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown compilation error"
        raise CompilationError(f"Compilation failed:\n{error_msg[:2000]}")

      if not exe_path.exists():
        raise CompilationError("Compilation produced no output")

      return exe_path

    finally:
      try:
        os.unlink(src_path)
      except Exception:
        pass

  def _compile_with_mono(self, source_code: str, exe_path: Path, extra_flags: List[str]) -> Path:
    """Compile using Mono mcs."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cs', delete=False) as f:
      f.write(source_code)
      src_path = f.name

    try:
      cmd = [
        self._compiler_path,
        '-optimize+',
        f'-out:{exe_path}',
        src_path,
      ] + extra_flags

      result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

      if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown compilation error"
        raise CompilationError(f"Compilation failed:\n{error_msg[:2000]}")

      if not exe_path.exists():
        raise CompilationError("Compilation produced no output")

      return exe_path

    finally:
      try:
        os.unlink(src_path)
      except Exception:
        pass

  def execute(self,
              exe_path: Path,
              stdin_data: str = "",
              timeout: float = 300,
              env: Dict[str, str] = None) -> Tuple[str, str, float, int]:
    """Execute compiled C# program."""
    start_time = time.time()
    process = None

    # For .NET DLLs, we need to run with dotnet
    if self._compiler_type == 'dotnet' and str(exe_path).endswith('.dll'):
      cmd = ['dotnet', str(exe_path)]
    elif self._compiler_type == 'mono' and self.is_linux:
      cmd = ['mono', str(exe_path)]
    else:
      cmd = [str(exe_path)]

    try:
      process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if self.is_windows else 0)

      stdout, stderr = process.communicate(input=stdin_data.encode('utf-8'), timeout=timeout)

      execution_time = time.time() - start_time
      return (stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace'),
              execution_time, process.returncode)

    except subprocess.TimeoutExpired:
      execution_time = time.time() - start_time
      if process:
        self._kill_process(process)
      raise ExecutionError(f"Execution timed out after {timeout} seconds")

    except Exception as e:
      execution_time = time.time() - start_time
      if process:
        self._kill_process(process)
      raise ExecutionError(f"Execution error: {str(e)}")

  def describe(self):
    """Return compiler flavour and version number."""
    compiler = self.find_compiler()
    if not compiler:
      raise CompilationError("C# compiler not found")

    sdk_dir_fp = ""
    if self._compiler_type == 'dotnet':
      try:
        sdk_dir = str((Path(compiler).resolve().parent / 'sdk'))
        sdk_dir_fp = _safe_stat_fingerprint(sdk_dir)
      except Exception:
        sdk_dir_fp = "unknown"

    cache_key = "|".join([
      platform.system(),
      platform.release(),
      platform.machine(),
      self._compiler_type or "unknown",
      compiler,
      _safe_stat_fingerprint(compiler),
      sdk_dir_fp,
    ])

    def max_langversion(lang_text: str) -> str:
      candidates = []
      for m in re.finditer(r"\b(\d+)(?:\.(\d+))?\b", lang_text or ""):
        major = int(m.group(1))
        minor = int(m.group(2)) if m.group(2) else 0
        candidates.append((major, minor))
      if not candidates:
        return "unknown"
      major, minor = max(candidates)
      return f"{major}.{minor}" if minor else f"{major}"

    def _compute() -> str:

      if self._compiler_type == 'dotnet':
        dotnet = compiler
        dotnet_ver = "unknown"
        try:
          r = subprocess.run([dotnet, '--version'], capture_output=True, text=True, timeout=10)
          if r.returncode == 0 and r.stdout.strip():
            dotnet_ver = r.stdout.strip()
        except Exception:
          pass

        sdk_ver = dotnet_ver
        try:
          r = subprocess.run([dotnet, '--list-sdks'], capture_output=True, text=True, timeout=10)
          if r.returncode == 0 and r.stdout.strip():

            def ver_key(v: str):
              main, _, pre = v.partition('-')
              parts = []
              for p in main.split('.'):
                try:
                  parts.append(int(p))
                except Exception:
                  parts.append(0)
              while len(parts) < 4:
                parts.append(0)
              is_stable = 1 if not pre else 0
              return (parts[0], parts[1], parts[2], parts[3], is_stable, pre)

            versions = []
            for line in r.stdout.splitlines():
              v = line.strip().split(' ', 1)[0]
              if v:
                versions.append(v)
            if versions:
              sdk_ver = sorted(versions, key=ver_key)[-1]
        except Exception:
          pass

        max_lv = "unknown"
        csc_ver = "unknown"
        try:
          dotnet_root = Path(dotnet).resolve().parent
          csc_dll = dotnet_root / 'sdk' / sdk_ver / 'Roslyn' / 'bincore' / 'csc.dll'
          if csc_dll.exists():
            r = subprocess.run([dotnet, str(csc_dll), '/langversion:?'],
                               capture_output=True,
                               text=True,
                               timeout=10)
            max_lv = max_langversion((r.stdout or "") + "\n" + (r.stderr or ""))

            rv = subprocess.run([dotnet, str(csc_dll), '/version'],
                                capture_output=True,
                                text=True,
                                timeout=10)
            if rv.stdout.strip():
              csc_ver = rv.stdout.strip().splitlines()[0].strip()
        except Exception:
          pass

        return f"C# (.NET SDK {dotnet_ver}; SDKs {sdk_ver}) | csc {csc_ver} | max LangVersion {max_lv}"

      if self._compiler_type == 'mono':
        try:
          r = subprocess.run([compiler, '--version'], capture_output=True, text=True, timeout=10)
          ver = r.stdout.strip() if r.stdout.strip() else "unknown"
        except Exception:
          ver = "unknown"
        return f"C# (Mono) | {ver}"

      if self._compiler_type == 'csc':
        csc = compiler
        ver = "unknown"
        max_lv = "unknown"
        try:
          rv = subprocess.run([csc, '/version'], capture_output=True, text=True, timeout=10)
          if rv.stdout.strip():
            ver = rv.stdout.strip().splitlines()[0].strip()
          rl = subprocess.run([csc, '/langversion:?'], capture_output=True, text=True, timeout=10)
          max_lv = max_langversion((rl.stdout or "") + "\n" + (rl.stderr or ""))
        except Exception:
          pass
        return f"C# (csc.exe {ver}) | max LangVersion {max_lv}"

      return "C# compiler | unknown"

    return _read_or_compute_description('compiler_csharp', cache_key, _compute)


def get_compiler(language: str, engine_name: str) -> NativeCompiler:
  """
    Factory function to get appropriate compiler.
    
    Args:
        language: 'cpp', 'rust', or 'csharp'
        engine_name: Name of the AI engine (for cache separation)
        
    Returns:
        Appropriate compiler instance
    """
  compilers = {
    'cpp': CppCompiler,
    'c++': CppCompiler,
    'rust': RustCompiler,
    'rs': RustCompiler,
    'csharp': CSharpCompiler,
    'c#': CSharpCompiler,
    'cs': CSharpCompiler,
  }

  language = language.lower()
  if language not in compilers:
    raise ValueError(f"Unsupported language: {language}")

  return compilers[language](engine_name)


def clean_cache(engine_name: str = None, language: str = None):
  """
    Clean compilation cache.
    
    Args:
        engine_name: Specific engine to clean (None for all)
        language: Specific language to clean (None for all)
    """
  if engine_name and language:
    target = COMPILE_CACHE_DIR / engine_name / language
  elif engine_name:
    target = COMPILE_CACHE_DIR / engine_name
  elif language:
    # Clean language across all engines
    if COMPILE_CACHE_DIR.exists():
      for engine_dir in COMPILE_CACHE_DIR.iterdir():
        lang_dir = engine_dir / language
        if lang_dir.exists():
          shutil.rmtree(lang_dir)
    return


def clear_compile_cache(engine_name: str = None):
  """Clear compiled executables from cache."""
  if engine_name:
    target = COMPILE_CACHE_DIR / engine_name
  else:
    target = COMPILE_CACHE_DIR

  if target.exists():
    shutil.rmtree(target)


class RunResult:
  """Result of a compile_and_run operation."""

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
    self.error_stage = error_stage  # 'compiler_missing', 'compilation', 'execution', 'timeout'

  def __bool__(self):
    return self.success

  def error_message(self, max_len: int = 200) -> str:
    """Get a formatted error message suitable for test output."""
    if self.success:
      return ""
    msg = self.error[:max_len] if self.error else self.stderr[:max_len]
    return msg


def compile_and_run(code: str,
                    language: str,
                    engine_name: str,
                    input_data: str = "",
                    input_file: Optional[Path] = None,
                    timeout: float = 300) -> RunResult:
  """
  Compile source code and run the resulting executable.
  
  This is the main helper function for test files - abstracts away all
  compiler detection, compilation, and execution handling.
  
  Args:
      code: Source code string
      language: Programming language ('cpp', 'rust', 'csharp')
      engine_name: AI engine name (for cache separation)
      input_data: String data to send to stdin (ignored if input_file provided)
      input_file: Path to file to use as stdin (for streaming large inputs)
      timeout: Execution timeout in seconds
      
  Returns:
      RunResult with success status, stdout, stderr, exec_time, and error info
  """
  try:
    compiler = get_compiler(language, engine_name)
  except ValueError as e:
    return RunResult(False, error=str(e), error_stage='compiler_missing')

  if not compiler.find_compiler():
    return RunResult(False, error=f"No {language} compiler found", error_stage='compiler_missing')

  try:
    exe_path = compiler.compile(code)
  except CompilationError as e:
    return RunResult(False, error=str(e), error_stage='compilation')

  try:
    if input_file is not None:
      stdout, stderr, exec_time, return_code = compiler.execute(exe_path,
                                                                timeout=timeout,
                                                                stdin_file=input_file)
    else:
      stdout, stderr, exec_time, return_code = compiler.execute(exe_path,
                                                                stdin_data=input_data,
                                                                timeout=timeout)

    success = return_code == 0
    error = stderr if not success else ""
    return RunResult(success,
                     stdout,
                     stderr,
                     exec_time,
                     return_code,
                     error,
                     error_stage='execution' if not success else '')

  except ExecutionError as e:
    return RunResult(False,
                     error=str(e),
                     error_stage='timeout' if 'timed out' in str(e) else 'execution')


# Test the compiler detection
if __name__ == "__main__":

  print(describe_this_pc())

  print("Testing compiler detection...")

  for lang, cls in [('cpp', CppCompiler), ('rust', RustCompiler), ('csharp', CSharpCompiler)]:
    compiler = cls("test_engine")
    path = compiler.find_compiler()
    if path:
      print(f"  {lang}: Found at {path}")
      if hasattr(compiler, '_compiler_type'):
        print(f"        Type: {compiler._compiler_type}")
      print(compiler.describe())
    else:
      print(f"  {lang}: Not found")
