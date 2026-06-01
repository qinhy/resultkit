@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem Build PyCUDA from source on Windows with CUDA/OpenGL enabled
rem using direct uv project Python.
rem
rem No uv venv.
rem No activation.
rem No --system.
rem ============================================================

rem ---- Project/source settings -------------------------------
set "PROJECT_DIR=%cd%"
set "WORKDIR=%PROJECT_DIR%\"
set "PYCUDA_REPO=https://github.com/inducer/pycuda.git"
set "PYCUDA_DIR=%WORKDIR%\pycuda"

rem uv run command used by your project.
set "UV_PYTHON_CMD=uv run python"

rem Avoid hardlink warning when uv cache and project are on different drives/filesystems.
set "UV_LINK_MODE=copy"
rem ------------------------------------------------------------

echo.
echo === PyCUDA OpenGL-enabled build with direct uv ===
echo PROJECT_DIR=%PROJECT_DIR%
echo WORKDIR=%WORKDIR%
echo UV_PYTHON_CMD=%UV_PYTHON_CMD%
echo.

rem ---- Check uv ----------------------------------------------
where uv.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv.exe not found on PATH.
    exit /b 1
)

uv --version
if errorlevel 1 exit /b 1

rem ---- Resolve real Python executable from uv ----------------
echo.
echo Resolving Python executable from: %UV_PYTHON_CMD%

for /f "usebackq delims=" %%i in (`%UV_PYTHON_CMD% -c "import sys; print(sys.executable)"`) do (
    set "PYTHON_EXE=%%i"
)

if "%PYTHON_EXE%"=="" (
    echo ERROR: Could not resolve Python executable from uv.
    exit /b 1
)

echo PYTHON_EXE=%PYTHON_EXE%
"%PYTHON_EXE%" -V
if errorlevel 1 exit /b 1

rem ---- Check CUDA --------------------------------------------
echo.
echo Checking CUDA...

if "%CUDA_PATH%"=="" (
    echo ERROR: CUDA_PATH is not set.
    echo Install NVIDIA CUDA Toolkit, then reopen this terminal.
    exit /b 1
)

if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo ERROR: nvcc.exe not found at "%CUDA_PATH%\bin\nvcc.exe"
    exit /b 1
)

echo CUDA_PATH=%CUDA_PATH%
"%CUDA_PATH%\bin\nvcc.exe" --version
if errorlevel 1 exit /b 1

rem ---- Load Visual Studio C++ build tools --------------------
echo.
echo Checking MSVC compiler...

where cl.exe >nul 2>nul
if errorlevel 1 (
    echo cl.exe not found. Trying to load Visual Studio Build Tools...

    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if not exist "!VSWHERE!" (
        echo ERROR: vswhere.exe not found.
        echo Install Visual Studio 2022 Build Tools with "Desktop development with C++".
        exit /b 1
    )

    for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VSINSTALL=%%i"
    )

    if "!VSINSTALL!"=="" (
        echo ERROR: Could not find Visual Studio C++ Build Tools.
        echo Install Visual Studio 2022 Build Tools with C++ support.
        exit /b 1
    )

    call "!VSINSTALL!\Common7\Tools\VsDevCmd.bat" -arch=x64
    if errorlevel 1 exit /b 1
)

where cl.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: cl.exe still not available after loading Visual Studio tools.
    exit /b 1
)

cl.exe 2>&1 | findstr /C:"Microsoft"

rem ---- Create build directory only ---------------------------
echo.
echo Preparing build directory...

if not exist "%WORKDIR%" mkdir "%WORKDIR%"
if errorlevel 1 exit /b 1

cd /d "%PROJECT_DIR%"
if errorlevel 1 exit /b 1

rem ---- Install build/runtime deps into uv-selected Python ----
echo.
echo Installing PyCUDA build dependencies into uv-selected Python...

uv pip install --python "%PYTHON_EXE%" ^
    setuptools ^
    wheel ^
    numpy ^
    mako ^
    pytools ^
    platformdirs ^
    pybind11 ^
    PyOpenGL ^
    PyOpenGL_accelerate

if errorlevel 1 (
    echo ERROR: Failed installing build dependencies.
    exit /b 1
)

rem ---- Clone or update PyCUDA source -------------------------
echo.
echo Fetching PyCUDA source...

if not exist "%PYCUDA_DIR%\.git" (
    git clone --recursive "%PYCUDA_REPO%" "%PYCUDA_DIR%"
    if errorlevel 1 exit /b 1
) else (
    cd /d "%PYCUDA_DIR%"
    if errorlevel 1 exit /b 1

    git pull
    if errorlevel 1 exit /b 1

    git submodule update --init --recursive
    if errorlevel 1 exit /b 1
)

cd /d "%PYCUDA_DIR%"
if errorlevel 1 exit /b 1

rem ---- Write PyCUDA siteconf.py with OpenGL enabled ----------
echo.
echo Writing PyCUDA siteconf.py with CUDA_ENABLE_GL=True...

> siteconf.py (
    echo BOOST_INC_DIR = []
    echo BOOST_LIB_DIR = []
    echo BOOST_COMPILER = ''
    echo USE_SHIPPED_BOOST = True
    echo BOOST_PYTHON_LIBNAME = []
    echo BOOST_THREAD_LIBNAME = []
    echo CUDA_TRACE = False
    echo CUDA_ROOT = r'%CUDA_PATH%'
    echo CUDA_PRETEND_VERSION = None
    echo CUDA_INC_DIR = [r'%CUDA_PATH%\include']
    echo CUDA_ENABLE_GL = True
    echo CUDA_ENABLE_CURAND = True
    echo CUDADRV_LIB_DIR = [r'%CUDA_PATH%\lib\x64']
    echo CUDADRV_LIBNAME = ['cuda']
    echo CUDART_LIB_DIR = [r'%CUDA_PATH%\lib\x64']
    echo CUDART_LIBNAME = ['cudart']
    echo CURAND_LIB_DIR = [r'%CUDA_PATH%\lib\x64']
    echo CURAND_LIBNAME = ['curand']
    echo CXXFLAGS = ['/EHsc']
    echo LDFLAGS = ['/FORCE']
)

type siteconf.py

rem ---- Configure PyCUDA --------------------------------------
echo.
echo Running PyCUDA configure.py...

"%PYTHON_EXE%" configure.py
if errorlevel 1 (
    echo ERROR: PyCUDA configure.py failed.
    exit /b 1
)

echo.
echo siteconf.py after configure:
type siteconf.py

rem ---- Build and install PyCUDA into uv-selected Python -------
echo.
echo Building and installing PyCUDA into uv-selected Python...

uv pip install --python "%PYTHON_EXE%" . --no-build-isolation --no-cache -v
if errorlevel 1 (
    echo ERROR: PyCUDA build/install failed.
    exit /b 1
)

rem ---- Verify ------------------------------------------------
echo.
echo Verifying PyCUDA and pycuda.gl...

"%PYTHON_EXE%" -c "import os; os.add_dll_directory(os.environ['CUDA_PATH'] + r'\bin'); import sys; import pycuda; import pycuda.driver as drv; import pycuda.gl as gl; print('Python:', sys.executable); print('PyCUDA OK:', pycuda.VERSION_TEXT); print('pycuda.gl OK:', gl)"
if errorlevel 1 (
    echo ERROR: PyCUDA import verification failed.
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS: PyCUDA was built with CUDA/OpenGL enabled.
echo Python used:
"%PYTHON_EXE%" -c "import sys; print(sys.executable)"
echo ============================================================

endlocal