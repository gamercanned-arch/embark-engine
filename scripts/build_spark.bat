@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist build mkdir build
cl /O2 /arch:AVX2 /fp:fast /EHsc /std:c++17 /F16777216 ^
   /Isrc\spark_engine ^
   /Isrc\spark_engine\opencl ^
   /I"include" ^
   src\spark_engine\main_spark_bench.cpp ^
   src\spark_engine\apu_spark_orchestrator.cpp ^
   src\spark_engine\spark_tiler.cpp ^
   src\spark_engine\spark_threadpool.cpp ^
   src\spark_engine\opencl\vega8_context.cpp ^
   src\spark_engine\opencl\OpenCL_sys.lib ^
   /Fe:build\spark_apu_engine.exe
