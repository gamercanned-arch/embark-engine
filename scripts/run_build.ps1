$proc = Start-Process cmd -ArgumentList '/c', 'cd /d C:\Users\abhik\Documents\antigravity\embark-engine && scripts\build_server.bat' -RedirectStandardOutput 'build\build_stdout.txt' -RedirectStandardError 'build\build_stderr.txt' -Wait -PassThru
Write-Host "Exit code: $($proc.ExitCode)"
Get-Content 'C:\Users\abhik\Documents\antigravity\embark-engine\build\build_stdout.txt' | Select-Object -Last 50
Get-Content 'C:\Users\abhik\Documents\antigravity\embark-engine\build\build_stderr.txt' | Select-Object -Last 20
