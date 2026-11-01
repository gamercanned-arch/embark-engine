@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist build mkdir build

cl /O2 /MD /arch:AVX2 /fp:fast /EHsc /std:c++17 /F16777216 ^
   /I"include" ^
   /I"src\server" ^
   /I"c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\include" ^
   /I"c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\ggml\include" ^
   src\server\main_server.cpp ^
   src\server\embark_vocab.cpp ^
   src\server\embark_sampler.cpp ^
   src\server\embark_session_manager.cpp ^
   src\server\embark_server.cpp ^
   "c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\build\src\llama.lib" ^
   "c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\build\ggml\src\ggml.lib" ^
   "c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\build\ggml\src\ggml-base.lib" ^
   "c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\build\ggml\src\ggml-cpu.lib" ^
   "c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\build\vendor\cpp-httplib\cpp-httplib.lib" ^
   ws2_32.lib crypt32.lib psapi.lib ^
   /Fe:build\embark_server.exe
