# venv на Python 3.11 + pip install (faiss-cpu не поддерживает Python 3.14 на Windows).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Создаю venv через: py -3.11 -m venv"
py -3.11 -m venv venv
if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Error "Не удалось создать venv. Установите Python 3.11 и проверьте: py -3.11 --version"
}

$pip = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
& $pip -m pip install --upgrade pip --default-timeout=120 --retries 10
& $pip -m pip install -r requirements.txt --default-timeout=180 --retries 10

Write-Host ""
Write-Host "Готово. Активация: .\venv\Scripts\Activate.ps1"
Write-Host "Заполните YANDEX_* в файле .env"
