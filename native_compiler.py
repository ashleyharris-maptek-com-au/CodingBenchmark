"""
Native Code Compiler Helpers

Provides helper classes for compiling C++, Rust, and C# code to native executables.
Supports Windows (Visual C++, MSVC, MinGW) and Linux (GCC, Clang) environments.
Includes caching to avoid recompilation when source hasn't changed.
"""

import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import time
import signal
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Base directory for compiled outputs
COMPILE_CACHE_DIR = Path(__file__).parent / "compile_cache"


def get_file_hash(content: str) -> str:
    """Generate hash of source code for caching."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


class CompilationError(Exception):
    """Raised when compilation fails."""
    pass


class ExecutionError(Exception):
    """Raised when execution fails."""
    pass


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
        self.cache_dir = ensure_dir(COMPILE_CACHE_DIR / self.engine_name /
                                    language)

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
                env: Dict[str, str] = None) -> Tuple[str, str, float, int]:
        """
        Execute compiled program.
        
        Args:
            exe_path: Path to executable
            stdin_data: Data to send to stdin
            timeout: Timeout in seconds
            env: Environment variables
            
        Returns:
            Tuple of (stdout, stderr, execution_time, return_code)
            
        Raises:
            ExecutionError: If execution fails or times out
        """
        start_time = time.time()
        process = None

        try:
            process = subprocess.Popen(
                [str(exe_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if self.is_windows else 0)

            stdout, stderr = process.communicate(
                input=stdin_data.encode('utf-8'), timeout=timeout)

            execution_time = time.time() - start_time
            return (stdout.decode('utf-8', errors='replace'),
                    stderr.decode('utf-8', errors='replace'), execution_time,
                    process.returncode)

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            if process:
                self._kill_process(process)
            raise ExecutionError(
                f"Execution timed out after {timeout} seconds")

        except Exception as e:
            execution_time = time.time() - start_time
            if process:
                self._kill_process(process)
            raise ExecutionError(f"Execution error: {str(e)}")

    def _kill_process(self, process: subprocess.Popen):
        """Kill process and all its children."""
        try:
            if self.is_windows:
                # Use taskkill to kill process tree on Windows
                subprocess.run(
                    ['taskkill', '/F', '/T', '/PID',
                     str(process.pid)],
                    capture_output=True,
                    timeout=5)
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
                        cl_path = os.path.join(vs_path, version, "bin",
                                               "Hostx64", "x64", "cl.exe")
                        if os.path.exists(cl_path):
                            return cl_path
                except Exception:
                    continue

        # Try vswhere
        try:
            vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
            if os.path.exists(vswhere):
                result = subprocess.run(
                    [vswhere, "-latest", "-property", "installationPath"],
                    capture_output=True,
                    text=True,
                    timeout=10)
                if result.returncode == 0:
                    vs_install = result.stdout.strip()
                    msvc_path = os.path.join(vs_install, "VC", "Tools", "MSVC")
                    if os.path.exists(msvc_path):
                        versions = sorted(os.listdir(msvc_path), reverse=True)
                        for version in versions:
                            cl_path = os.path.join(msvc_path, version, "bin",
                                                   "Hostx64", "x64", "cl.exe")
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
                result = subprocess.run(
                    [vswhere, "-latest", "-property", "installationPath"],
                    capture_output=True,
                    text=True,
                    timeout=10)
                if result.returncode == 0:
                    vs_install = result.stdout.strip()
                    vcvars = os.path.join(vs_install, "VC", "Auxiliary",
                                          "Build", "vcvars64.bat")
                    if os.path.exists(vcvars):
                        # Run vcvars64.bat and capture environment
                        cmd = f'cmd /c ""{vcvars}" && set"'
                        result = subprocess.run(cmd,
                                                capture_output=True,
                                                text=True,
                                                shell=True,
                                                timeout=30)
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp',
                                         delete=False) as f:
            f.write(source_code)
            src_path = f.name

        try:
            if self._compiler_type == 'msvc':
                # MSVC compilation
                env = self._get_msvc_env()
                obj_path = str(exe_path.with_suffix('.obj'))
                cmd = [
                    compiler,
                    '/nologo',
                    '/EHsc',
                    '/O2',
                    '/std:c++17',
                    src_path,
                    f'/Fe:{exe_path}',
                    f'/Fo:{obj_path}',
                ] + extra_flags

                result = subprocess.run(cmd,
                                        capture_output=True,
                                        text=True,
                                        env=env,
                                        timeout=120)

                # Clean up obj file
                try:
                    os.unlink(obj_path)
                except Exception:
                    pass

            else:
                # GCC/Clang/MinGW compilation
                cmd = [
                    compiler,
                    '-O2',
                    '-std=c++17',
                    '-o',
                    str(exe_path),
                    src_path,
                ] + extra_flags

                if self.is_linux:
                    cmd.append('-pthread')

                result = subprocess.run(cmd,
                                        capture_output=True,
                                        text=True,
                                        timeout=120)

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown compilation error"
                raise CompilationError(
                    f"Compilation failed:\n{error_msg[:2000]}")

            if not exe_path.exists():
                raise CompilationError("Compilation produced no output")

            return exe_path

        finally:
            try:
                os.unlink(src_path)
            except Exception:
                pass


class RustCompiler(NativeCompiler):
    """Rust compiler using rustc or cargo."""

    def __init__(self, engine_name: str):
        super().__init__(engine_name, "rust")
        self._compiler_path = None

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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rs',
                                         delete=False) as f:
            f.write(source_code)
            src_path = f.name

        try:
            cmd = [
                compiler,
                '-O',
                '--edition',
                '2021',
                '-o',
                str(exe_path),
                src_path,
            ] + extra_flags

            result = subprocess.run(cmd,
                                    capture_output=True,
                                    text=True,
                                    timeout=120)

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown compilation error"
                raise CompilationError(
                    f"Compilation failed:\n{error_msg[:2000]}")

            if not exe_path.exists():
                raise CompilationError("Compilation produced no output")

            return exe_path

        finally:
            try:
                os.unlink(src_path)
            except Exception:
                pass


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
            dotnet = shutil.which('dotnet')
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
            dotnet = shutil.which('dotnet')
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

    def _find_csc(self) -> Optional[str]:
        """Find csc.exe from .NET Framework or Roslyn."""
        # .NET Framework paths
        framework_paths = [
            r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
            r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe",
        ]

        for path in framework_paths:
            if os.path.exists(path):
                return path

        # Try to find Roslyn csc
        try:
            vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
            if os.path.exists(vswhere):
                result = subprocess.run(
                    [vswhere, "-latest", "-property", "installationPath"],
                    capture_output=True,
                    text=True,
                    timeout=10)
                if result.returncode == 0:
                    vs_install = result.stdout.strip()
                    roslyn_csc = os.path.join(vs_install, "MSBuild", "Current",
                                              "Bin", "Roslyn", "csc.exe")
                    if os.path.exists(roslyn_csc):
                        return roslyn_csc
        except Exception:
            pass

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
            raise CompilationError(
                "No C# compiler found (dotnet, csc, or mcs)")

        extra_flags = extra_flags or []
        exe_path = self._get_cached_exe_path(source_hash)

        if self._compiler_type == 'dotnet':
            return self._compile_with_dotnet(source_code, source_hash,
                                             exe_path, extra_flags)
        elif self._compiler_type == 'csc':
            return self._compile_with_csc(source_code, exe_path, extra_flags)
        else:  # mono
            return self._compile_with_mono(source_code, exe_path, extra_flags)

    def _compile_with_dotnet(self, source_code: str, source_hash: str,
                             exe_path: Path, extra_flags: List[str]) -> Path:
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
            result = subprocess.run([
                self._compiler_path, 'build', '-c', 'Release', '-o',
                str(self.cache_dir)
            ],
                                    cwd=str(project_dir),
                                    capture_output=True,
                                    text=True,
                                    timeout=120)

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown compilation error"
                raise CompilationError(
                    f"Compilation failed:\n{error_msg[:2000]}")

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

    def _compile_with_csc(self, source_code: str, exe_path: Path,
                          extra_flags: List[str]) -> Path:
        """Compile using csc.exe."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cs',
                                         delete=False) as f:
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

            result = subprocess.run(cmd,
                                    capture_output=True,
                                    text=True,
                                    timeout=120)

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown compilation error"
                raise CompilationError(
                    f"Compilation failed:\n{error_msg[:2000]}")

            if not exe_path.exists():
                raise CompilationError("Compilation produced no output")

            return exe_path

        finally:
            try:
                os.unlink(src_path)
            except Exception:
                pass

    def _compile_with_mono(self, source_code: str, exe_path: Path,
                           extra_flags: List[str]) -> Path:
        """Compile using Mono mcs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cs',
                                         delete=False) as f:
            f.write(source_code)
            src_path = f.name

        try:
            cmd = [
                self._compiler_path,
                '-optimize+',
                f'-out:{exe_path}',
                src_path,
            ] + extra_flags

            result = subprocess.run(cmd,
                                    capture_output=True,
                                    text=True,
                                    timeout=120)

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown compilation error"
                raise CompilationError(
                    f"Compilation failed:\n{error_msg[:2000]}")

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
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                if self.is_windows else 0)

            stdout, stderr = process.communicate(
                input=stdin_data.encode('utf-8'), timeout=timeout)

            execution_time = time.time() - start_time
            return (stdout.decode('utf-8', errors='replace'),
                    stderr.decode('utf-8', errors='replace'), execution_time,
                    process.returncode)

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            if process:
                self._kill_process(process)
            raise ExecutionError(
                f"Execution timed out after {timeout} seconds")

        except Exception as e:
            execution_time = time.time() - start_time
            if process:
                self._kill_process(process)
            raise ExecutionError(f"Execution error: {str(e)}")


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
    else:
        target = COMPILE_CACHE_DIR

    if target.exists():
        shutil.rmtree(target)


# Test the compiler detection
if __name__ == "__main__":
    print("Testing compiler detection...")

    for lang, cls in [('cpp', CppCompiler), ('rust', RustCompiler),
                      ('csharp', CSharpCompiler)]:
        compiler = cls("test_engine")
        path = compiler.find_compiler()
        if path:
            print(f"  {lang}: Found at {path}")
            if hasattr(compiler, '_compiler_type'):
                print(f"        Type: {compiler._compiler_type}")
        else:
            print(f"  {lang}: Not found")
