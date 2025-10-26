chcp 65001 > $null

git config i18n.commitEncoding utf-8
git config i18n.logOutputEncoding utf-8

Write-Host "=== pull origin main ==="
git pull origin main
Write-Host ""

Write-Host "=== git status ==="
git status
Write-Host ""

Write-Host "=== git add . ==="
git add .
Write-Host ""

$message = Read-Host "commit message (please type in Chinese or English)"

if (-not $message -or $message.Trim() -eq "") {
    Write-Host "empty message, abort."
    exit
}

Write-Host "=== git commit ==="
git commit -m $message
Write-Host ""

Write-Host "=== git push origin main ==="
git push origin main
Write-Host ""

Write-Host "upload done."
