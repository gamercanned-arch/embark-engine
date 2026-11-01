@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist build mkdir build
cl /O2 /arch:AVX2 /fp:fast /EHsc /std:c++17 /F16777216 ^
   /Isrc\embedding_gemma_engine ^
   /Isrc\embedding_gemma_engine\opencl ^
   /I"include" ^
   src\embedding_gemma_engine\main_embedding_bench.cpp ^
   src\embedding_gemma_engine\apu_embed_orchestrator.cpp ^
   src\embedding_gemma_engine\embed_tiler.cpp ^
   src\embedding_gemma_engine\embed_threadpool.cpp ^
   src\embedding_gemma_engine\opencl\vega8_context.cpp ^
   src\embedding_gemma_engine\opencl\OpenCL_sys.lib ^
   /Fe:build\embedding_gemma_apu_engine.exe
