# Forward Python arguments to this task's GPU environment.
$ErrorActionPreference = 'Stop'
$taskWorkspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$taskPython = Join-Path $taskWorkspace 'work\.venv-gpu\Scripts\python.exe'
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
