@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist build mkdir build
cl /O2 /arch:AVX2 /fp:fast /EHsc /std:c++17 /F16777216 ^
   /Isrc\gemma4_e4b_engine ^
   /Isrc\gemma4_e4b_engine\opencl ^
   /I"include" ^
   src\gemma4_e4b_engine\main_gemma4_e4b_bench.cpp ^
   src\gemma4_e4b_engine\apu_gemma4_e4b_orchestrator.cpp ^
   src\gemma4_e4b_engine\gemma4_e4b_tiler.cpp ^
   src\gemma4_e4b_engine\gemma4_e4b_threadpool.cpp ^
   src\gemma4_e4b_engine\opencl\vega8_context.cpp ^
   src\gemma4_e4b_engine\opencl\OpenCL_sys.lib ^
   /Fe:build\gemma4_e4b_apu_engine.exe
