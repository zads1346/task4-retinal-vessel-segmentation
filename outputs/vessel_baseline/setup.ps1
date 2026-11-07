param(
    [string]$BasePython = 'C:\Users\lenovo\anaconda3\envs\dome\python.exe',
    [switch]$Standalone
)
$ErrorActionPreference = 'Stop'
$taskWorkspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$taskEnv = Join-Path $taskWorkspace 'work\.venv-gpu'
$taskPython = Join-Path $taskEnv 'Scripts\python.exe'
if ($Standalone -and (Test-Path -LiteralPath (Join-Path $taskEnv 'pyvenv.cfg'))) {
    $taskVenvConfig = Get-Content -LiteralPath (Join-Path $taskEnv 'pyvenv.cfg') -Raw
    if ($taskVenvConfig -match 'include-system-site-packages\s*=\s*true') {
        throw 'This task environment intentionally inherits CUDA packages. Standalone creation requires a fresh environment path; existing files were left intact.'
    }
}
if (-not (Test-Path -LiteralPath $taskPython)) {
    if ($Standalone) {
        $taskDrive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($taskEnv))
        if ($taskDrive.AvailableFreeSpace -lt 12GB) {
            throw 'Standalone CUDA install needs at least 12 GB free for wheels and unpacking. Reuse the existing CUDA environment or free space first.'
        }
        python -m venv $taskEnv
    } else {
        if (-not (Test-Path -LiteralPath $BasePython)) {
            throw 'Existing CUDA Python not found. Set -BasePython to a verified CUDA environment, or use -Standalone after ensuring disk space.'
        }
        & $BasePython -c "import torch,torchvision; assert torch.cuda.is_available(); print(torch.__version__,torchvision.__version__)"
        if ($LASTEXITCODE -ne 0) { throw 'Base CUDA environment check failed' }
        & $BasePython -m venv --system-site-packages $taskEnv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
}
if ($Standalone) {
    & $taskPython -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
}
& $taskPython -m pip install --no-cache-dir -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $taskPython (Join-Path $PSScriptRoot 'check_environment.py') --output (Join-Path $PSScriptRoot 'environment.json')
if ($LASTEXITCODE -ne 0) { throw 'GPU validation failed' }
