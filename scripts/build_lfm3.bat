@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist build mkdir build
cl /O2 /arch:AVX2 /fp:fast /EHsc /std:c++17 /F16777216 ^
   /Isrc\lfm3_vl_engine ^
   /Isrc\lfm3_vl_engine\opencl ^
   /I"include" ^
   src\lfm3_vl_engine\main_lfm3_bench.cpp ^
   src\lfm3_vl_engine\apu_lfm3_orchestrator.cpp ^
   src\lfm3_vl_engine\lfm3_tiler.cpp ^
   src\lfm3_vl_engine\lfm3_threadpool.cpp ^
   src\lfm3_vl_engine\opencl\vega8_context.cpp ^
   src\lfm3_vl_engine\opencl\OpenCL_sys.lib ^
   /Fe:build\lfm3_apu_engine.exe
