param(
    [string]$DataZip = 'C:\Users\lenovo\Downloads\2026task-4.zip',
    [int]$Fold = 0,
    [int]$Steps = 6000,
    [string]$RunName = 'fold0',
    [switch]$Resume,
    [switch]$SkipTraining
)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName may contain letters, digits, underscore, dash only' }
$taskWorkspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$taskPython = Join-Path $taskWorkspace 'work\.venv-gpu\Scripts\python.exe'
$taskData = Join-Path $taskWorkspace 'work\data'
$taskRun = Join-Path $PSScriptRoot ('runs\' + $RunName)
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run .\setup.ps1 first' }
& $taskPython (Join-Path $PSScriptRoot 'prepare.py') --zip $DataZip --destination $taskData --report (Join-Path $PSScriptRoot 'data_audit.json')
if ($LASTEXITCODE -ne 0) { throw 'Dataset integrity check failed' }
if (-not $SkipTraining) {
    $taskTrainArgs = @((Join-Path $PSScriptRoot 'train.py'), '--data', $taskData, '--output', $taskRun,
                       '--fold', "$Fold", '--steps', "$Steps", '--batch-size', '8')
    if ($Resume) { $taskTrainArgs += @('--resume', (Join-Path $taskRun 'last.pt')) }
    & $taskPython @taskTrainArgs
    if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect its last output' }
}
$taskReport = Get-Content -LiteralPath (Join-Path $taskRun 'validation_report.json') -Raw | ConvertFrom-Json
$taskThreshold = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0}', $taskReport.threshold_tuning.native_best_threshold)
$taskUseTta = $false
$taskSelectionPath = Join-Path $PSScriptRoot 'selection.json'
if (Test-Path -LiteralPath $taskSelectionPath) {
    $taskSelection = Get-Content -LiteralPath $taskSelectionPath -Raw | ConvertFrom-Json
    if ($taskSelection.checkpoint -eq [System.IO.Path]::GetFullPath((Join-Path $taskRun 'best.pt'))) {
        $taskCheckpointHash = (Get-FileHash -LiteralPath (Join-Path $taskRun 'best.pt') -Algorithm SHA256).Hash
        if ($taskCheckpointHash -ne $taskSelection.checkpoint_sha256) {
            throw 'Selected inference configuration belongs to a different checkpoint. Revalidate before regenerating submission.'
        }
        $taskThreshold = [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0}', $taskSelection.threshold)
        $taskUseTta = [bool]$taskSelection.tta
    }
}
$taskPredictArgs = @((Join-Path $PSScriptRoot 'predict.py'), '--data', $taskData, '--checkpoint',
                    (Join-Path $taskRun 'best.pt'), '--output', (Join-Path $taskRun 'submission'),
                    '--threshold', $taskThreshold, '--polarity', 'vessel-white', '--resize-mode', 'official')
if ($taskUseTta) { $taskPredictArgs += '--tta' }
& $taskPython @taskPredictArgs
if ($LASTEXITCODE -ne 0) { throw 'Prediction or CSV verification failed' }
Write-Output ('Ready: ' + (Join-Path $taskRun 'submission\submission.csv'))
Write-Output 'CSV encodes white vessels using the original conversion. Hidden ground-truth polarity/resize remains unconfirmed; no upload was performed.'
