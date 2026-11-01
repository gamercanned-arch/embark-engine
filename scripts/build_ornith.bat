@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist build mkdir build
cl /O2 /arch:AVX2 /fp:fast /EHsc /std:c++17 /F16777216 ^
   /Isrc\ornith_engine ^
   /Isrc\ornith_engine\opencl ^
   /I"include" ^
   src\ornith_engine\main_ornith_bench.cpp ^
   src\ornith_engine\apu_ornith_orchestrator.cpp ^
   src\ornith_engine\ornith_tiler.cpp ^
   src\ornith_engine\ornith_threadpool.cpp ^
   src\ornith_engine\opencl\vega8_context.cpp ^
   src\ornith_engine\opencl\OpenCL_sys.lib ^
   /Fe:build\ornith_apu_engine.exe
