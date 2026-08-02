# Run the PARMBench Workflows v1 comparison ladder.
#
# Every condition gets the same model, environment, tools, corpus, retrieval
# index, per-observation budget, and step limit. Only the memory policy changes.
#
# -Tier picks the corpus scale point. -Samples runs each condition that many
# times with independent trajectories, because a single run of an agent does not
# distinguish a policy effect from ordinary run-to-run variance.

param(
    [string]$Python = 'C:\Users\karth\anaconda3\python.exe',
    [string]$Dataset = 'data\workflows_v1',
    [ValidateSet('tier-28', 'tier-100')]
    [string]$Tier = 'tier-100',
    [int]$Samples = 3,
    [string]$Results = 'data\benchmark-results\workflows-v1',
    [string]$Caches = 'data\workflow-caches',
    [string]$Model = 'gpt-5-mini',
    [int]$Workers = 3
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'src'

$indexes = @{
    'tier-28'  = 'data\retrieval-indexes\workflow-eng-lead-v1'
    'tier-100' = 'data\retrieval-indexes\workflow-eng-lead-v1-100'
}
$Index = $indexes[$Tier]
$outputRoot = Join-Path $Results $Tier
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$trajectoryCache = Join-Path $Caches 'trajectories-gpt5mini'
# The admission cache namespace is derived from the retrieval index hash, so a
# tier change cannot silently replay the other tier's admissions.
$admissionCache = Join-Path $Caches "parm-admission-$Tier"

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
    foreach ($sample in 0..($Samples - 1)) {
        $out = Join-Path $outputRoot "$name.sample$sample.jsonl"
        Write-Host "== $Tier $name sample $sample =="
        & $Python -m parm_bench.cli workflow run $Dataset `
            --policy $name `
            --model $Model `
            --trajectory-cache $trajectoryCache `
            --sample $sample `
            --workers $Workers `
            --out $out `
            @($condition.args)
        if ($LASTEXITCODE -ne 0) { throw "workflow run failed for $name sample $sample" }
        & $Python -m parm_bench.cli workflow score $out `
            --gold $Dataset `
            --out (Join-Path $outputRoot "$name.sample$sample.metrics.json")
        if ($LASTEXITCODE -ne 0) { throw "workflow score failed for $name sample $sample" }
    }
}

Write-Host "`nDone. Results in $outputRoot"
