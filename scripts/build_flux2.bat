@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist build mkdir build
cl /O2 /arch:AVX2 /fp:fast /EHsc /std:c++17 /F16777216 ^
   /Isrc\flux2_dit_engine ^
   /Isrc\flux2_dit_engine\opencl ^
   /I"include" ^
   src\flux2_dit_engine\main_flux2_bench.cpp ^
   src\flux2_dit_engine\apu_flux2_orchestrator.cpp ^
   src\flux2_dit_engine\flux2_tiler.cpp ^
   src\flux2_dit_engine\flux2_threadpool.cpp ^
   src\flux2_dit_engine\opencl\vega8_context.cpp ^
   src\flux2_dit_engine\opencl\OpenCL_sys.lib ^
   /Fe:build\flux2_apu_engine.exe
