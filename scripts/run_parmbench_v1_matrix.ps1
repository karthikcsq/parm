param(
    [string]$Model = 'gpt-5-mini'
)

# First-pass baseline ladder over the frozen benchmark_parmbench_v1 batch.
# Every condition shares the same dataset, response model, and frozen
# 300-persona retrieval index. Predictions, metrics, and config sidecars land
# under one first-pass namespace and must not be regenerated after results
# have been inspected. The no_memory rung replays the fairness response cache,
# so it makes no live answer calls.

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = 'C:\Users\karth\anaconda3\python.exe'
$Dataset = Join-Path $RepoRoot 'data\benchmark_parmbench_v1'
$Index = Join-Path $RepoRoot 'data\retrieval-indexes\personamem-v2-train-v1'
$Results = Join-Path $RepoRoot 'data\benchmark-results\parmbench-v1-first-pass'
$Responses = Join-Path $RepoRoot 'data\response-caches\parmbench-v1-first-pass'
$FairnessCache = Join-Path $RepoRoot 'data\response-caches\parmbench-v1-fairness'
$Expansions = Join-Path $RepoRoot 'data\expansion-caches\parmbench-v1-first-pass'
$Admissions = Join-Path $RepoRoot 'data\parm-admission-caches\parmbench-v1-first-pass'
$ModelSlug = $Model -replace '[^A-Za-z0-9._-]', '-'
$env:PYTHONPATH = Join-Path $RepoRoot 'src'

function Invoke-ParmBench {
    param([string[]]$Arguments)

    & $Python -m parm_bench.cli @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "parm-bench failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Condition {
    param(
        [string]$Name,
        [string[]]$RunArguments,
        [string]$ResponseCache,
        [switch]$NoIndex
    )

    $Predictions = Join-Path $Results "$Name.jsonl"
    $Metrics = Join-Path $Results "$Name.metrics.json"
    $Config = Join-Path $Results "$Name.config.json"
    if (-not $ResponseCache) {
        $ResponseCache = Join-Path $Responses $Name
    }
    if (
        -not (Test-Path -LiteralPath $Predictions) -or
        -not (Test-Path -LiteralPath $Config)
    ) {
        $Arguments = @('run', $Dataset)
        if (-not $NoIndex) {
            $Arguments += @(
                '--retrieval-index', $Index,
                '--retrieval-limit', '5'
            )
        }
        $Arguments += @(
            '--response-cache', $ResponseCache,
            '--response-policy', 'populate',
            '--model', $Model,
            '--out', $Predictions
        ) + $RunArguments
        Invoke-ParmBench $Arguments
    }
    Invoke-ParmBench @(
        'score',
        $Predictions,
        '--gold', $Dataset,
        '--out', $Metrics
    )
    Write-Output "condition-complete: $Name"
}

Invoke-Condition "no-memory-$ModelSlug" @(
    '--baseline', 'no_memory'
) -ResponseCache $FairnessCache -NoIndex

Invoke-Condition "input-rag-enhanced-$ModelSlug" @(
    '--baseline', 'input_rag',
    '--retrieval-mode', 'enhanced',
    '--expansion-cache', $Expansions,
    '--expansion-policy', 'populate'
)

Invoke-Condition "naive-output-tool-then-model-hybrid-$ModelSlug" @(
    '--baseline', 'naive_output_rag',
    '--output-rag-flow', 'tool_then_model_output',
    '--retrieval-mode', 'hybrid'
)

Invoke-Condition "all-entity-output-rag-$ModelSlug" @(
    '--baseline', 'all_entity_output_rag'
)

Invoke-Condition "prompted-memory-tool-hybrid-$ModelSlug" @(
    '--baseline', 'prompted_memory_tool',
    '--retrieval-mode', 'hybrid'
)

Invoke-Condition "parm-convergence-$ModelSlug" @(
    '--baseline', 'parm',
    '--parm-retriever', 'convergence'
)

Invoke-Condition "parm-semantic-judge-$ModelSlug" @(
    '--baseline', 'parm',
    '--parm-retriever', 'semantic-judge',
    '--parm-admission-cache', $Admissions,
    '--parm-admission-policy', 'populate'
)

Write-Output 'matrix-complete'
