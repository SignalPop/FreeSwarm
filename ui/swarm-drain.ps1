<#
  Drain the swarm runner: ask it to stop starting new work, then wait until it exits on its own.
  Also the one place that finds, kills and counts swarm runners (-Hard, -Check).

  Used by ui\restart-swarm.cmd, start-services.cmd and stop-services.cmd. The runner
  (swarm_runner.py) honours ui\backend\.swarm_drain: while the file exists no agent starts an
  iteration, chore, mentor pass or task; what is running finishes normally; once nothing runs the
  runner deletes the file and exits 0 (after FREESWARM_DRAIN_MAX_S, default 5400 s, it exits
  anyway and logs what it cut). This script creates the file and waits, printing the runner's
  progress from ui\backend\.swarm_drain.status.json.

  Modes:
    (default)     drain and wait.
    -CloseWindow  also close every runner window ("cmd /k run-swarm.bat") -- FIRST, before the
                  runner exits, then sweep any runner still alive once the drain is over.
    -Hard         no drain: close every runner window, then kill every runner process tree now.
    -Check        exit 1 (and say so) if a runner is running, else 0. Changes nothing.

  Why windows go first (10-01 20:32, two runners after `restart-swarm.cmd --hard`): cmd runs a
  batch file by re-reading it from disk after each command, resuming at the old byte offset.
  run-swarm.bat had been edited while the runner ran, so when the runner was killed its window
  resumed mid-file in the new text and started a second runner. run-swarm.bat now parses its
  launch as one block, but windows started from an older copy are still exposed -- so the host
  cmd is closed while the runner still holds it, and a cmd that is gone cannot re-run anything.

  Each runner is TWO python.exe processes: the venv's python.exe is a launcher that runs the
  script in a child base interpreter, both with swarm_runner.py on their command line. Both are
  matched, and each is killed with its process tree.

  Exit codes: 0 the runner exited (or was not running); 2 it outlived the max wait plus margin
  and was killed; 3 it never acknowledged the drain (a runner started before drain support --
  restart it once with restart-swarm.cmd --hard), and it was left running; 4 a runner survived
  -Hard or the final sweep. -Check: 1 a runner is running.
  Never prompts, so it can run unattended.
#>
param(
  [int]$MaxWaitS = 0,          # 0: FREESWARM_DRAIN_MAX_S, else 5400
  [int]$MarginS = 300,         # on top of the max wait, for the runner's own exit
  [int]$AckS = 90,             # a runner with drain support writes its status within seconds
  [switch]$CloseWindow,        # close the runner's "cmd /k run-swarm.bat" window(s) as well
  [switch]$Hard,               # kill now instead of draining (closes the windows too)
  [switch]$Check,              # only report whether a runner is running (exit 1 if so)
  # The runner's command line and its window's command line (regexes). The environment
  # overrides exist for the script test, which runs a fake runner under another name.
  [string]$Match = $(if ($env:FREESWARM_RUNNER_MATCH) { $env:FREESWARM_RUNNER_MATCH } else { 'swarm_runner\.py' }),
  [string]$WindowMatch = $(if ($env:FREESWARM_RUNNER_WINDOW_MATCH) { $env:FREESWARM_RUNNER_WINDOW_MATCH } else { 'run-swarm\.bat' })
)

$ErrorActionPreference = 'Stop'
$backend = Join-Path $PSScriptRoot 'backend'
$drainFile = if ($env:FREESWARM_DRAIN_FILE) { $env:FREESWARM_DRAIN_FILE } else { Join-Path $backend '.swarm_drain' }
$statusFile = "$drainFile.status.json"
if ($MaxWaitS -le 0) {
  $MaxWaitS = 5400
  if ($env:FREESWARM_DRAIN_MAX_S) { $MaxWaitS = [int][double]$env:FREESWARM_DRAIN_MAX_S }
}

function Say([string]$msg) { Write-Host ("   [{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg) }

function Get-Runner {
  @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -match $Match })
}

# This script's own ancestors: restart-swarm.cmd may itself have been typed into an old runner
# window (its runner already gone), and closing that window would kill the restart half-way.
function Get-Ancestors {
  $seen = @{}
  $id = $PID
  for ($i = 0; $i -lt 32; $i++) {
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$id" -ErrorAction SilentlyContinue
    if (-not $p -or $seen.ContainsKey([int]$p.ProcessId)) { break }
    $seen[[int]$p.ProcessId] = $true
    $id = $p.ParentProcessId
  }
  $seen
}
$ancestors = Get-Ancestors

# Every runner window: the hosts of running runners AND stale ones whose runner already exited
# (17:09 on 10-01 -- a --hard restart used to kill only python and leave the window open).
function Get-RunnerWindow {
  @(Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -match $WindowMatch -and -not $ancestors.ContainsKey([int]$_.ProcessId) })
}

# Closes only the cmd.exe, not its tree: a runner still running keeps its console (and keeps
# draining); the point is that nothing is left to re-run run-swarm.bat when the runner ends.
function Close-RunnerWindows {
  foreach ($w in Get-RunnerWindow) {
    Stop-Process -Id $w.ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host ("   [close] swarm runner window (cmd pid {0})" -f $w.ProcessId)
  }
}

function Stop-Tree([int]$Id) {
  $kids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$Id" -ErrorAction SilentlyContinue)
  Stop-Process -Id $Id -Force -ErrorAction SilentlyContinue
  foreach ($k in $kids) { Stop-Tree ([int]$k.ProcessId) }
}

# Kill every runner (launcher and child), then re-scan until none is left: windows first each
# round, so a cmd that resumed its batch in between is caught along with what it started.
function Stop-AllRunners {
  for ($round = 0; $round -lt 6; $round++) {
    Close-RunnerWindows
    $procs = Get-Runner
    if ($procs.Count -eq 0) { return $true }
    foreach ($p in $procs) {
      if (Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue) {
        Write-Host ("   [kill] swarm runner (pid {0})" -f $p.ProcessId)
        Stop-Tree ([int]$p.ProcessId)
      }
    }
    Start-Sleep -Milliseconds 1500
  }
  (Get-Runner).Count -eq 0
}

function Remove-DrainFiles {
  foreach ($f in @($drainFile, $statusFile)) { Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue }
}

if ($Check) {
  $procs = Get-Runner
  if ($procs.Count -eq 0) { exit 0 }
  Write-Host ("   [note] a swarm runner is already running (pid {0})" -f (($procs | ForEach-Object { $_.ProcessId }) -join ', '))
  exit 1
}

if ($Hard) {
  if ((Get-Runner).Count -eq 0) { Write-Host '   [ -- ] swarm runner not running' }
  $ok = Stop-AllRunners
  Remove-DrainFiles
  if (-not $ok) {
    Write-Host ("   [X] a swarm runner is still running (pid {0})" -f ((Get-Runner | ForEach-Object { $_.ProcessId }) -join ', '))
    exit 4
  }
  exit 0
}

$procs = Get-Runner
if ($procs.Count -eq 0) {
  Say 'swarm runner not running - nothing to drain'
  Remove-DrainFiles
  if ($CloseWindow) { Close-RunnerWindows }
  exit 0
}
$pids = @($procs | ForEach-Object { [int]$_.ProcessId })

# Before the runner can exit: see the header.
if ($CloseWindow) { Close-RunnerWindows }

Set-Content -LiteralPath $drainFile -Value ("drain requested {0} by {1}" -f (Get-Date -Format s), $env:USERNAME) -Encoding ascii
Say ("drain requested (pid {0}); waiting up to {1} min for running iterations to finish" -f ($pids -join ', '), [math]::Round(($MaxWaitS + $MarginS) / 60))

$t0 = Get-Date
$acked = $false
$lastLine = ''
$lastPrint = [datetime]::MinValue
$rc = 0
while ($true) {
  $alive = @($pids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
  if ($alive.Count -eq 0) { break }
  $elapsed = ((Get-Date) - $t0).TotalSeconds

  $status = $null
  if (Test-Path -LiteralPath $statusFile) {
    $acked = $true
    try { $status = Get-Content -LiteralPath $statusFile -Raw | ConvertFrom-Json } catch { $status = $null }
  } elseif (-not (Test-Path -LiteralPath $drainFile)) {
    $acked = $true   # the runner removed the request: it is on its way out
  }

  if (-not $acked -and $elapsed -ge $AckS) {
    Say "the runner has not acknowledged the drain after $AckS s - it predates drain support."
    Say 'drain request withdrawn; the runner is still running. Restart it once with:  ui\restart-swarm.cmd --hard'
    Remove-DrainFiles
    exit 3
  }
  if ($elapsed -ge $MaxWaitS + $MarginS) {
    Say ("still running after {0} min - killing it" -f [math]::Round($elapsed / 60))
    $alive | ForEach-Object { Stop-Tree ([int]$_) }
    $rc = 2
    break
  }

  if ($status) {
    $line = "{0} still running" -f $status.running
    if ($line -ne $lastLine -or ((Get-Date) - $lastPrint).TotalSeconds -ge 60) {
      Say ("{0} ({1} min in)" -f $line, [math]::Round($elapsed / 60))
      foreach ($it in @($status.iterations)) {
        if ($it) {
          $mins = [math]::Round(([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [double]$it.since) / 60)
          Write-Host ("          - {0}: {1} ({2} min)" -f $it.agent, $it.work, $mins)
        }
      }
      $lastLine = $line
      $lastPrint = Get-Date
    }
  } elseif (((Get-Date) - $lastPrint).TotalSeconds -ge 60) {
    Say ("waiting for the runner ({0} min in)" -f [math]::Round($elapsed / 60))
    $lastPrint = Get-Date
  }
  Start-Sleep -Seconds 5
}

Remove-DrainFiles
Say ("swarm runner exited after {0} min" -f [math]::Round(((Get-Date) - $t0).TotalSeconds / 60, 1))
if ($CloseWindow) {
  # A restart follows: nothing else may be left running -- a runner some other window started
  # meanwhile, or a second one that was already running beside the drained one.
  if ((Get-Runner).Count -gt 0) { Say 'another swarm runner is still running - killing it' }
  if (-not (Stop-AllRunners)) {
    Say ("a swarm runner is still running (pid {0})" -f ((Get-Runner | ForEach-Object { $_.ProcessId }) -join ', '))
    exit 4
  }
}
exit $rc
