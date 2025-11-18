# format-code.ps1

Write-Host "🚀 Iniciando formateo automático con Black..." -ForegroundColor Green

# 1. Verificar que Black esté instalado
if (!(Get-Command black -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Black no está instalado. Instalando..." -ForegroundColor Yellow
    pip install black
}

# 2. Buscar y formatear archivos .py (excluyendo venv/)
Write-Host "🔍 Buscando archivos Python..." -ForegroundColor Cyan

$pyFiles = Get-ChildItem -Recurse -Name "*.py" | Where-Object { 
    $_.FullName -notlike "*venv*" -and 
    $_.FullName -notlike "*__pycache__*" -and
    $_.FullName -notlike "*.git*"
}

foreach ($file in $pyFiles) {
    Write-Host "   🛠️ Formateando: $file" -ForegroundColor Gray
}

# 3. Ejecutar Black en todos los archivos
Write-Host "🔧 Aplicando formato con Black..." -ForegroundColor Cyan
black $pyFiles

# 4. Verificar que no haya más cambios pendientes
Write-Host "✅ Verificando formato final..." -ForegroundColor Green
$checkResult = black --check $pyFiles

if ($LASTEXITCODE -eq 0) {
    Write-Host "🎉 ¡Todo está correctamente formateado!" -ForegroundColor Green
} else {
    Write-Host "❌ Error: Aún hay archivos sin formatear." -ForegroundColor Red
    exit 1
}

# 5. Verificar si hay cambios pendientes de commit
$gitStatus = git status --porcelain
$hasPyChanges = $gitStatus | Select-String -Pattern "\.py"

if ($hasPyChanges) {
    Write-Host "📝 Committing cambios de formato..." -ForegroundColor Yellow
    git add .
    git commit -m "chore: format all Python files with Black" --no-verify
    Write-Host "✅ Commit generado: 'chore: format all Python files with Black'" -ForegroundColor Green

    # Preguntar si hacer push
    $response = Read-Host "¿Deseas hacer push a GitHub? (y/N)"
    if ($response -match "^[Yy]$") {
        git push
        Write-Host "📤 Push realizado con éxito." -ForegroundColor Green
    } else {
        Write-Host "⏭️ Omitido push. Recuerda hacerlo manualmente después." -ForegroundColor Yellow
    }
} else {
    Write-Host "ℹ️ No hay cambios pendientes de commit (todo ya estaba formateado)." -ForegroundColor Gray
}

Write-Host "✨ Formateo completado con éxito." -ForegroundColor Green