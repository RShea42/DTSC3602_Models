# Start both local apps, each in its own window:   powershell -File start_local.ps1
#   http://127.0.0.1:8000   demo page with Steam sign-in (Juan's page, repaired) - local only, never deployed
#   http://127.0.0.1:8001   model API, the same code as on Modal; docs at /docs
# Python: $env:STEAMREC_PYTHON if set, else "python" (use the environment made from requirements.txt).
# Steam key (demo page only): $env:STEAM_API_KEY, else STEAM_API_KEY=... from .env here or in ..\Steam_EASE_Safe_Recommender
$here = $PSScriptRoot
$py = if ($env:STEAMREC_PYTHON) { $env:STEAMREC_PYTHON } else { "python" }
$key = $env:STEAM_API_KEY
foreach ($f in @((Join-Path $here ".env"), (Join-Path $here "..\Steam_EASE_Safe_Recommender\.env"))) {
  if (-not $key -and (Test-Path $f)) { $key = (Get-Content $f | Where-Object { $_ -like "STEAM_API_KEY=*" }) -replace "^STEAM_API_KEY=", "" }
}
if (-not $key) { Write-Host "No STEAM_API_KEY found: the demo page will load, but Steam lookups will fail." }
Start-Process -WorkingDirectory $here -FilePath "powershell" -ArgumentList "-NoExit", "-Command",
  "`$env:PYTHONUTF8='1'; `$env:STEAM_API_KEY='$key'; `$env:PUBLIC_BASE_URL='http://127.0.0.1:8000'; `$env:SESSION_BACKEND='memory'; & '$py' -m uvicorn app.main:app --port 8000"
Start-Process -WorkingDirectory $here -FilePath "powershell" -ArgumentList "-NoExit", "-Command",
  "`$env:PYTHONUTF8='1'; & '$py' -m uvicorn app.api:app --port 8001"
Write-Host "Demo page: http://127.0.0.1:8000   Model API: http://127.0.0.1:8001/docs"
