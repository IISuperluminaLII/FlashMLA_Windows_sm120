@echo off
echo [OK] Setting environment for SM120 build...
set FLASH_MLA_DISABLE_SM100=0
set FLASH_MLA_DISABLE_SM90=1
set FLASH_MLA_SM120_DISABLE_BWD=0

echo [OK] Environment variables:
echo   FLASH_MLA_DISABLE_SM100=%FLASH_MLA_DISABLE_SM100%
echo   FLASH_MLA_DISABLE_SM90=%FLASH_MLA_DISABLE_SM90%
echo   FLASH_MLA_SM120_DISABLE_BWD=%FLASH_MLA_SM120_DISABLE_BWD%

echo [OK] Cleaning build directory...
rd /s /q build 2>nul
del /q *.pyd 2>nul

echo [OK] Starting build...
"C:\Users\Shashank Murthy\.conda\envs\150BLLM\python.exe" setup.py build_ext --inplace