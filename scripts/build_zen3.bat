@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist build mkdir build
cl /O2 /arch:AVX2 /fp:fast /EHsc /std:c++17 /F16777216 ^
   /Isrc\zen3_engine ^
   /Isrc\zen3_engine\opencl ^
   /I"include" ^
   src\zen3_engine\main_apu_bench.cpp ^
   src\zen3_engine\apu_orchestrator.cpp ^
   src\zen3_engine\lfm2_tiler.cpp ^
   src\zen3_engine\zen3_threadpool.cpp ^
   src\zen3_engine\opencl\vega8_context.cpp ^
   src\zen3_engine\opencl\OpenCL_sys.lib ^
   /Fe:build\zen3_apu_engine.exe
