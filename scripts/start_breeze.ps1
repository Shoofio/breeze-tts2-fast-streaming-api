#Requires -Version 5.1
<#
.SYNOPSIS
    Starts the Breeze TTS streaming API natively on Windows.

.DESCRIPTION
    Windows counterpart to scripts/start_breeze.sh, which runs under WSL.

    The repo's .venv is a Linux virtualenv built under WSL and cannot be used
    from Windows, so this script maintains a separate .venv-win and leaves the
    WSL environment completely untouched.

    On first run it bootstraps .venv-win: CUDA-enabled torch from the PyTorch
    index first, then the remaining dependencies from requirements.txt, then
    triton-windows (needed because the --fast-all depth decoder calls
    torch.compile and PyTorch ships no Triton for Windows). Later runs skip
    straight to launching unless requirements.txt or the Triton pin has
    changed.

.PARAMETER BindHost
    Interface to bind. Defaults to 0.0.0.0 (matches start_breeze.sh); this
    exposes the port to the LAN and Windows Firewall will prompt on first
    launch. Use 127.0.0.1 to keep it local.

.PARAMETER Port
    Port to listen on. Defaults to 8080.

.PARAMETER Cors
    CORS mode for browser clients (e.g. SillyTavern). Empty (the default)
    leaves CORS off, so no browser origin is allowed. Pass '*' to allow every
    origin -- this is how to send the API's bare `--cors` flag from
    PowerShell, since a string parameter can't be given with no value -- or a
    comma-separated allowlist, e.g.
    -Cors 'http://127.0.0.1:8000,http://localhost:8000'.

.PARAMETER WsPort
    WebSocket port. Defaults to the HTTP port + 1. Pass 'disabled' to turn
    the WebSocket endpoint off.

.PARAMETER ModelPath
    Override the checkpoint path. By default the current HuggingFace snapshot
    is resolved from the hub cache's refs/main.

.PARAMETER NoFastAll
    Run the eager path (~7.7 GiB VRAM) instead of the CUDA-graph fast path
    (~14.4 GiB). Use this if CUDA graph capture misbehaves.

.PARAMETER AttnImplementation
    Attention kernel for the backbone and text encoder: eager or sdpa.
    Defaults to eager. On an RTX 4090, sdpa was no faster but used about
    2.7 GB less peak VRAM with the fast path.

.PARAMETER SkipSetup
    Skip the environment check entirely and launch immediately.

.PARAMETER Reinstall
    Delete .venv-win and rebuild it from scratch.

.PARAMETER ExtraArgs
    Additional arguments forwarded verbatim to breeze_infer.api.

.EXAMPLE
    .\scripts\start_breeze.ps1

.EXAMPLE
    .\scripts\start_breeze.ps1 -BindHost 127.0.0.1 -NoFastAll

.EXAMPLE
    .\scripts\start_breeze.ps1 -AttnImplementation sdpa
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$BindHost = '0.0.0.0',
    [int]$Port = 8080,
    [string]$Cors = '',
    [string]$WsPort,
    [string]$ModelPath,
    [switch]$NoFastAll,
    [ValidateSet('eager', 'sdpa')]
    [string]$AttnImplementation = 'eager',
    [switch]$SkipSetup,
    [switch]$Reinstall,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

# --- Paths -----------------------------------------------------------------
# $Host is a read-only PowerShell automatic variable, which is why the bind
# parameter above is named -BindHost rather than -Host.

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$VenvDir    = Join-Path $RepoRoot '.venv-win'
$VenvPy     = Join-Path $VenvDir 'Scripts\python.exe'
$ReqFile    = Join-Path $RepoRoot 'requirements.txt'
$StampFile  = Join-Path $VenvDir '.breeze-deps.sha256'
$HubDir     = '$HF_HOME\hub\models--BreezeBlue--Breeze-TTS-2'
$TorchIndex = 'https://download.pytorch.org/whl/cu128'
$PyVersion  = '3.12'
# PyTorch ships no Triton for Windows, but --fast-all's depth-decoder stage
# calls torch.compile, whose inductor backend needs it. triton-windows is the
# community Windows build; 3.5.x is the series matching torch 2.9. It compiles
# via MSVC, which it locates from the Visual Studio install itself.
$TritonPkg  = 'triton-windows==3.5.1.post24'

function Invoke-Step {
    param([string]$Description, [scriptblock]$Command)
    Write-Host "==> $Description" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed (exit code $LASTEXITCODE)."
    }
}

# breeze_infer and models are imported as top-level packages, so the server
# must run from the repo root. Push/Pop so the caller's location survives.
Push-Location $RepoRoot
try {
    # --- Resolve the checkpoint --------------------------------------------
    if (-not $ModelPath) {
        $refFile = Join-Path $HubDir 'refs\main'
        if (-not (Test-Path -LiteralPath $refFile)) {
            throw "HuggingFace ref not found: $refFile`nPass -ModelPath to point at the checkpoint directly."
        }
        # refs/main has no trailing newline, but trim defensively anyway.
        $sha = (Get-Content -Raw -LiteralPath $refFile).Trim()
        if (-not $sha) { throw "HuggingFace ref is empty: $refFile" }
        $ModelPath = Join-Path $HubDir "snapshots\$sha"
    }
    if (-not (Test-Path -LiteralPath $ModelPath)) {
        throw "Model snapshot not found: $ModelPath"
    }

    # --- Bootstrap the Windows virtualenv ----------------------------------
    if ($Reinstall -and (Test-Path -LiteralPath $VenvDir)) {
        Write-Host "==> Removing $VenvDir" -ForegroundColor Cyan
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }

    if (-not $SkipSetup) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw "uv was not found on PATH. Install it from https://astral.sh/uv or run with -SkipSetup once .venv-win exists."
        }
        if (-not (Test-Path -LiteralPath $ReqFile)) {
            throw "requirements.txt not found at $ReqFile"
        }

        if (-not (Test-Path -LiteralPath $VenvPy)) {
            Invoke-Step "Creating $VenvDir (Python $PyVersion)" {
                uv venv --python $PyVersion $VenvDir
            }
        }

        # Only reinstall when requirements.txt actually changes, so a normal
        # launch touches the network not at all. The stamp also covers the
        # Windows-only Triton pin: it lives outside requirements.txt, so
        # bumping it here must re-trigger setup too.
        $reqFileHash = (Get-FileHash -LiteralPath $ReqFile -Algorithm SHA256).Hash
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $stampBytes = [System.Text.Encoding]::UTF8.GetBytes($reqFileHash + $TritonPkg)
            $reqHash = [System.BitConverter]::ToString($sha256.ComputeHash($stampBytes)) -replace '-', ''
        } finally {
            $sha256.Dispose()
        }
        $haveHash = ''
        if (Test-Path -LiteralPath $StampFile) {
            $haveHash = (Get-Content -Raw -LiteralPath $StampFile).Trim()
        }

        if ($haveHash -ne $reqHash) {
            # Order matters. CUDA torch must land first: the second install
            # then sees 2.9.1+cu128 as already satisfying `torch==2.9.1`
            # (PEP 440 ignores the local version segment for a bare ==) and
            # leaves it alone. Reversing these pulls PyPI's CPU-only Windows
            # wheel and the CUDA graph fast path dies at capture.
            Invoke-Step "Installing CUDA torch + torchaudio from $TorchIndex" {
                uv pip install --python $VenvPy --index-url $TorchIndex torch==2.9.1 torchaudio==2.9.1
            }
            Invoke-Step "Installing remaining dependencies from requirements.txt" {
                uv pip install --python $VenvPy -r $ReqFile
            }
            # Windows-only, so it stays out of the shared requirements.txt.
            # Installed from PyPI, not the PyTorch index used above.
            Invoke-Step "Installing $TritonPkg (Windows Triton for torch.compile)" {
                uv pip install --python $VenvPy $TritonPkg
            }

            Write-Host "==> Verifying CUDA availability" -ForegroundColor Cyan
            & $VenvPy -c "import torch; from torch.utils._triton import has_triton; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| triton', has_triton())"
            if ($LASTEXITCODE -ne 0) { throw "torch failed to import from $VenvPy." }

            Set-Content -LiteralPath $StampFile -Value $reqHash -Encoding ascii
        }
    }

    if (-not (Test-Path -LiteralPath $VenvPy)) {
        throw "Python interpreter not found at $VenvPy. Re-run without -SkipSetup to build the environment."
    }

    # --- Launch ------------------------------------------------------------
    # The server emits one JSON object per line on stdout; keep it unbuffered
    # so those events stream live instead of sitting in a block buffer.
    $env:PYTHONUNBUFFERED = '1'

    # torch 2.9's static CUDA launcher assumes a 64-bit C long and overflows on
    # Windows ("Python int too large to convert to C long") the first time an
    # inductor-compiled kernel launches. Falling back to the standard launcher
    # is the supported escape hatch and costs only a little launch overhead.
    $env:TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER = '0'

    # Keep the inductor cache inside .venv-win. Otherwise it inherits whatever
    # TORCHINDUCTOR_CACHE_DIR is already set in the shell (on this machine an
    # unrelated tool points it at a shared .unsloth directory), which mixes
    # artifacts from different torch builds.
    $env:TORCHINDUCTOR_CACHE_DIR = Join-Path $VenvDir 'inductor-cache'

    if ($NoFastAll) { $fastFlag = '--no-fast-all' } else { $fastFlag = '--fast-all' }

    $argList = @(
        '-m', 'breeze_infer.api',
        $ModelPath,
        '--host', $BindHost,
        '--port', $Port,
        $fastFlag,
        '--attn-implementation', $AttnImplementation
    )
    if ($Cors) { $argList += @('--cors', $Cors) }
    if ($WsPort) { $argList += @('--ws-port', $WsPort) }
    if ($ExtraArgs) { $argList += $ExtraArgs }

    Write-Host "==> Serving $ModelPath on http://${BindHost}:$Port ($fastFlag, attn $AttnImplementation)" -ForegroundColor Green
    & $VenvPy @argList
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
