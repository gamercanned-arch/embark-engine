@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /O2 /MD /EHsc /std:c++17 ^
   /I"include" ^
   scratch\test_server.cpp ^
   "c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\build\vendor\cpp-httplib\cpp-httplib.lib" ^
   ws2_32.lib crypt32.lib ^
   /Fe:scratch\test_server.exe
