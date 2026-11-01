@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /O2 /EHsc /std:c++17 ^
   /I"c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\include" ^
   /I"c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\ggml\include" ^
   scratch\test_vocab_load.cpp ^
   "c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\build\src\llama.lib" ^
   "c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\build\ggml\src\ggml.lib" ^
   "c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\build\ggml\src\ggml-base.lib" ^
   "c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\build\ggml\src\ggml-cpu.lib" ^
   /Fe:scratch\test_vocab_load.exe
