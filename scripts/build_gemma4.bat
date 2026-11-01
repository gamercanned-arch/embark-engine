@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist build mkdir build
cl /O2 /arch:AVX2 /fp:fast /EHsc /std:c++17 /F16777216 ^
   /Isrc\gemma4_engine ^
   /Isrc\gemma4_engine\opencl ^
   /I"include" ^
   src\gemma4_engine\main_gemma4_bench.cpp ^
   src\gemma4_engine\apu_gemma4_orchestrator.cpp ^
   src\gemma4_engine\gemma4_tiler.cpp ^
   src\gemma4_engine\gemma4_threadpool.cpp ^
   src\gemma4_engine\opencl\vega8_context.cpp ^
   src\gemma4_engine\opencl\OpenCL_sys.lib ^
   /Fe:build\gemma4_apu_engine.exe
