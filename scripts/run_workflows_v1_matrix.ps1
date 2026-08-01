# Run the PARMBench Workflows v1 comparison ladder.
#
# Every condition gets the same model, environment, tools, corpus, retrieval
# index, and per-observation budget. Only the memory policy changes.

param(
    [string]$Python = 'C:\Users\karth\anaconda3\python.exe',
    [string]$Dataset = 'data\workflows_v1',
    [string]$Index = 'data\retrieval-indexes\workflow-eng-lead-v1',
    [string]$Results = 'data\benchmark-results\workflows-v1',
    [string]$Caches = 'data\workflow-caches',
    [string]$Model = 'gpt-5-mini',
    [int]$Workers = 3
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'
New-Item -ItemType Directory -Force -Path $Results | Out-Null

$trajectoryCache = Join-Path $Caches 'trajectories-gpt5mini'
$admissionCache = Join-Path $Caches 'parm-admission-v1'

$conditions = @(
    @{ name = 'no_memory'; args = @() },
    @{ name = 'input_rag'; args = @('--retrieval-index', $Index, '--retrieval-mode', 'dense') },
    @{ name = 'naive_output_rag'; args = @('--retrieval-index', $Index, '--retrieval-mode', 'dense') },
    @{ name = 'all_entity_output_rag'; args = @('--retrieval-index', $Index) },
    @{ name = 'prompted_memory_tool'; args = @('--retrieval-index', $Index, '--retrieval-mode', 'dense') },
    @{ name = 'parm'; args = @('--retrieval-index', $Index, '--parm-admission-cache', $admissionCache) }
)

foreach ($condition in $conditions) {
    $name = $condition.name
    $out = Join-Path $Results "$name.jsonl"
    Write-Host "== $name =="
    & $Python -m parm_bench.cli workflow run $Dataset `
        --policy $name `
        --model $Model `
        --trajectory-cache $trajectoryCache `
        --workers $Workers `
        --out $out `
        @($condition.args)
    if ($LASTEXITCODE -ne 0) { throw "workflow run failed for $name" }
    & $Python -m parm_bench.cli workflow score $out `
        --gold $Dataset `
        --out (Join-Path $Results "$name.metrics.json")
    if ($LASTEXITCODE -ne 0) { throw "workflow score failed for $name" }
}

Write-Host "`nDone. Results in $Results"
