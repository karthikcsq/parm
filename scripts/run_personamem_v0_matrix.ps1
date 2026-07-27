param(
    [string]$Model = 'gpt-5-mini'
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = 'C:\Users\karth\anaconda3\python.exe'
$Dataset = Join-Path $RepoRoot 'data\benchmark_personamem_v0'
$Index = Join-Path $RepoRoot 'data\retrieval-indexes\personamem-v2-train-v0'
$Results = Join-Path $RepoRoot 'data\benchmark-results\personamem-v0-first-pass'
$Responses = Join-Path $RepoRoot 'data\response-caches\personamem-v0-first-pass'
$Expansions = Join-Path $RepoRoot 'data\expansion-caches\personamem-v0-first-pass'
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
        [string[]]$RunArguments
    )

    $Predictions = Join-Path $Results "$Name.jsonl"
    $Metrics = Join-Path $Results "$Name.metrics.json"
    $Config = Join-Path $Results "$Name.config.json"
    if (
        -not (Test-Path -LiteralPath $Predictions) -or
        -not (Test-Path -LiteralPath $Config)
    ) {
        $Arguments = @(
            'run',
            $Dataset,
            '--retrieval-index', $Index,
            '--retrieval-limit', '5',
            '--response-cache', (Join-Path $Responses $Name),
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
}

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
Invoke-Condition "parm-$ModelSlug" @(
    '--baseline', 'parm'
)
