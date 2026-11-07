# Forward Python arguments to this task's GPU environment.
$ErrorActionPreference = 'Stop'
$taskWorkspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$taskPython = Join-Path $taskWorkspace 'work\.venv-gpu\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = Join-Path $taskWorkspace 'work\v7_temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'GPU environment missing. Run .\setup.ps1 first.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $taskPython @args
    $taskExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExitCode
