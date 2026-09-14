@echo off
chcp 65001 > nul
echo ========================================================
echo        Запуск Argent Mobile для Android
echo ========================================================
echo.

set LOCAL_IP=
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Test-Connection -ComputerName (hostname) -Count 1).IPV4Address.IPAddressToString"`) do set LOCAL_IP=%%i
if "%LOCAL_IP%"=="" set LOCAL_IP=localhost

echo [✓] Компьютер (localhost):  http://localhost:3000
echo [✓] Смартфон в сети Wi-Fi:   http://%LOCAL_IP%:3000
echo.
echo Подключите телефон к тому же Wi-Fi и откройте адрес выше.
echo В меню браузера Chrome нажмите «Добавить на главный экран».
echo.
npm run dev
pause
