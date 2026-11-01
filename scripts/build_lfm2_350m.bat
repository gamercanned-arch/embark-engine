@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist build mkdir build
cl /O2 /arch:AVX2 /fp:fast /EHsc /std:c++17 /F16777216 ^
   /Isrc\lfm2_350m_engine ^
   /Isrc\lfm2_350m_engine\opencl ^
   /I"include" ^
   src\lfm2_350m_engine\main_lfm2_350m_bench.cpp ^
   src\lfm2_350m_engine\apu_lfm2_orchestrator.cpp ^
   src\lfm2_350m_engine\lfm2_tiler.cpp ^
   src\lfm2_350m_engine\lfm2_threadpool.cpp ^
   src\lfm2_350m_engine\opencl\vega8_context.cpp ^
   src\lfm2_350m_engine\opencl\OpenCL_sys.lib ^
   /Fe:build\lfm2_350m_apu_engine.exe
