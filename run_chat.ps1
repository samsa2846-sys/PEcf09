# Запуск консоли с UTF-8 (чтобы кириллица в вопросах не ломалась в PowerShell).
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot
.\venv\Scripts\python.exe app.py
