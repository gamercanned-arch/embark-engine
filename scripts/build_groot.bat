@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist build mkdir build
cl /O2 /arch:AVX2 /fp:fast /EHsc /std:c++17 /F16777216 ^
   /Isrc\groot_engine ^
   /Isrc\groot_engine\opencl ^
   /I"include" ^
   src\groot_engine\main_groot_bench.cpp ^
   src\groot_engine\apu_groot_orchestrator.cpp ^
   src\groot_engine\groot_tiler.cpp ^
   src\groot_engine\groot_threadpool.cpp ^
   src\groot_engine\opencl\vega8_context.cpp ^
   src\groot_engine\opencl\OpenCL_sys.lib ^
   /Fe:build\groot_apu_engine.exe
