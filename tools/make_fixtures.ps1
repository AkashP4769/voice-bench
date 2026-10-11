<#
.SYNOPSIS
    Generate speech fixtures for the benchmark tests.

.DESCRIPTION
    Uses the Windows speech synthesizer to produce 16 kHz mono PCM WAV files,
    which is exactly the input both engines expect. Synthesized speech has a
    known transcript, so it doubles as the reference text for word error rate.

    Windows only; the fast test suite generates its own synthetic audio and does
    not depend on these files.

.EXAMPLE
    powershell -File tools/make_fixtures.ps1
#>

[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path (Split-Path -Parent $PSScriptRoot) "tests\fixtures")
)

Add-Type -AssemblyName System.Speech

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

function Save-Speech {
    param([string]$Text, [string]$File)
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.Rate = 0
    $synth.SetOutputToWaveFile($File, $format)
    $synth.Speak($Text)
    $synth.Dispose()
}

$sentence = "The quick brown fox jumps over the lazy dog."

# A short clip for accuracy and latency checks.
Save-Speech -Text $sentence -File (Join-Path $OutputDirectory "short_en.wav")

# Repeated past 30 s, to show that the engines segment long audio themselves.
$long = (($sentence + " ") * 14).Trim()
Save-Speech -Text $long -File (Join-Path $OutputDirectory "long_en.wav")

Write-Host "reference text for short_en.wav:"
Write-Host "  $sentence"

foreach ($file in Get-ChildItem $OutputDirectory -Filter *.wav) {
    # 16 kHz mono 16-bit is 32 bytes per millisecond.
    $milliseconds = [math]::Round($file.Length / 32.0)
    Write-Host ("{0,-16} {1,9} bytes  ~{2,7} ms" -f $file.Name, $file.Length, $milliseconds)
}
