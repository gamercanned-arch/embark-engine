@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist build mkdir build
cl /O2 /arch:AVX2 /fp:fast /EHsc /std:c++17 /F16777216 ^
   /Isrc\d1_engine ^
   /Isrc\d1_engine\opencl ^
   /I"include" ^
   src\d1_engine\main_d1_bench.cpp ^
   src\d1_engine\apu_d1_orchestrator.cpp ^
   src\d1_engine\d1_tiler.cpp ^
   src\d1_engine\d1_threadpool.cpp ^
   src\d1_engine\opencl\vega8_context.cpp ^
   src\d1_engine\opencl\OpenCL_sys.lib ^
   /Fe:build\d1_apu_engine.exe
