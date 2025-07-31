@echo off
setlocal

echo This script will help you initialize and push the CURRENT folder as a Git repository to GitHub.
echo.
echo IMPORTANT PRE-REQUISITES:
echo 1. You must have Git installed and configured on your system.
echo 2. You MUST have already created an EMPTY repository on GitHub with the EXACT SAME NAME
echo    as this local folder.
echo 3. Ensure you have a GitHub Personal Access Token (PAT) configured for Git,
echo    or be ready to enter your GitHub credentials when prompted during the push.
echo.

set /p "github_username=Enter your GitHub username: "

echo.
echo Starting the push process for the current folder...
echo.

rem Get the full path of the current directory
set "current_dir=%CD%"

rem Get the name of the current folder, which we assume is also the GitHub repository name
for %%i in ("%current_dir%") do set "repo_name=%%~nxi"

rem Construct the GitHub repository URL
set "github_url=https://github.com/%github_username%/%repo_name%.git"

echo Processing folder: "%current_dir%"

rem Check if the current directory contains a .git folder (indicating it's already a Git repository)
if not exist "%current_dir%\.git" (
    echo Initializing new Git repository in "%current_dir%"...
    git init
    if errorlevel 1 (
        echo Error: Failed to initialize Git repository. Exiting.
        goto :eof
    )
    echo Adding all files to the repository...
    git add .
    echo Creating initial commit...
    git commit -m "Initial commit"
    if errorlevel 1 (
        echo Warning: Initial commit failed. This might be because there are no files to commit.
        echo Please add files and commit manually if needed before pushing.
    )
) else (
    echo A Git repository already exists in "%current_dir%".
    echo If you proceed, the existing Git history will be DELETED and a new repository will be created.
    set /p "confirm_reinit=Are you sure you want to delete the existing .git folder and re-initialize? (yes/no): "
    if /i "%confirm_reinit%"=="yes" (
        echo Deleting existing .git folder...
        rmdir /s /q "%current_dir%\.git"
        if errorlevel 1 (
            echo Error: Failed to delete existing .git folder. Please delete it manually and try again. Exiting.
            goto :eof
        )
        echo Re-initializing new Git repository in "%current_dir%"...
        git init
        if errorlevel 1 (
            echo Error: Failed to re-initialize Git repository. Exiting.
            goto :eof
        )
        echo Adding all files to the repository...
        git add .
        echo Creating initial commit...
        git commit -m "Initial commit (re-initialized)"
        if errorlevel 1 (
            echo Warning: Initial commit failed after re-initialization. This might be because there are no files to commit.
            echo Please add files and commit manually if needed before pushing.
        )
    ) else (
        echo Operation cancelled. No changes made to the Git repository.
        echo Press any key to exit.
        pause > nul
        goto :eof
    )
)

echo Checking if remote 'origin' exists for this repository...
rem Check if 'origin' remote is already configured
git remote -v | findstr /i "origin" > nul
if %errorlevel% neq 0 (
    rem If 'origin' does not exist, add it
    echo Adding remote 'origin': %github_url%
    git remote add origin %github_url%
    if %errorlevel% neq 0 (
        echo Error: Failed to add remote 'origin' for "%current_dir%". Please check the URL and permissions.
        echo.
        goto :eof
    )
) else (
    rem If 'origin' exists, update its URL to ensure it points to the correct GitHub repo
    echo Remote 'origin' already exists. Updating its URL to: %github_url%
    git remote set-url origin %github_url%
    if %errorlevel% neq 0 (
        echo Error: Failed to update remote 'origin' for "%current_dir%".
        echo.
        goto :eof
    )
)

echo Pushing local changes to GitHub...
rem Push to the 'main' branch first, then fall back to 'master' if 'main' doesn't exist
git push -u origin main || git push -u origin master
if %errorlevel% neq 0 (
    echo Warning: Push failed for "%current_dir%". Please review the error messages above and resolve manually.
) else (
    echo Successfully pushed "%current_dir%" to GitHub.
)
echo.

echo All done!
echo Press any key to exit.
pause > nul
endlocal
