# Run by Heliox's uninstaller (installer-hooks.nsh, NSIS_HOOK_PREUNINSTALL) before it removes the app's files. It
# ends the app installed at $env:HELIOX_APP, then the daemon that Heliox's own environment runs, each with its process
# tree. The app goes first, so its supervisor cannot start the daemon again. Without this the uninstaller's own check
# ended only the app, abruptly, and the daemon went on running from the person's environment. Only the person's own
# processes can be ended; nothing is deleted here.
$ErrorActionPreference = 'SilentlyContinue'

function Stop-Tree([int] $id) {
    & taskkill.exe /F /T /PID $id | Out-Null
}

$app = $env:HELIOX_APP
if ($app) {
    Get-CimInstance Win32_Process |
        Where-Object { $_.ExecutablePath -and ($_.ExecutablePath -ieq $app) } |
        ForEach-Object { Stop-Tree $_.ProcessId }
    Start-Sleep -Milliseconds 500
}
# The daemon: "<home>\.config\heliox-os\env\Scripts\python.exe -m pilot.server", or the private Python under
# "<home>\.config\heliox-os\python" that the environment's launcher started. Only Heliox's own environment folder
# counts: a developer's daemon from a source checkout (often cloned as "Heliox-OS") is not the installed app's
# (end_heliox_check.py runs this script against both).
$environment = '*\.config\heliox-os\env\*'
$private = '*\.config\heliox-os\python\*'
Get-CimInstance Win32_Process |
    Where-Object {
        $_.CommandLine -like '*-m pilot.server*' -and (
            $_.CommandLine -like $environment -or $_.CommandLine -like $private -or $_.ExecutablePath -like $private
        )
    } |
    ForEach-Object { Stop-Tree $_.ProcessId }
exit 0
