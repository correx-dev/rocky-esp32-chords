# Instala todo lo necesario para Rocky en Windows (PowerShell). Ejecutar desde la carpeta del proyecto:
#   powershell -ExecutionPolicy Bypass -File instalar_herramientas.ps1
# NOTA: escrito sin poder ejecutarlo en tu equipo; si un paso falla, copia el comando y corre ese paso a mano.

Write-Host "1/4 Librerias de Python"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "2/4 Arduino CLI (compilar y cargar el firmware)"
if (-not (Get-Command arduino-cli -ErrorAction SilentlyContinue)) {
    winget install --id ArduinoSA.CLI -e --accept-package-agreements --accept-source-agreements
    Write-Host "Cierra y abre PowerShell si 'arduino-cli' no se reconoce todavia."
}

Write-Host "3/4 Placa ESP32 y librerias del firmware"
arduino-cli config init --overwrite
arduino-cli config add board_manager.additional_urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli lib install "Chirale_TensorFlowLite"
arduino-cli lib install "Adafruit INA219"     # solo si mediras energia (USAR_INA219 1 en config.h)

Write-Host "4/4 Pruebas rapidas (sin hardware)"
python -m unittest tests/test_estructuras.py
Write-Host "Listo. Siguiente: lee docs/GUIA_PASO_A_PASO.md"
