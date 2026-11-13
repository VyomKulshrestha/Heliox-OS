; Heliox's English text for the Windows installer and uninstaller (bundle.windows.nsis.customLanguageFiles). Tauri
; uses this file instead of its own English one, so every string its installer script names is defined here, in
; Heliox's words; tests/test_windows_installer_text.py checks that the list matches the names Tauri's script uses
; (TAURI_STRINGS there) and that the two strings below say what they do.
;
; deleteAppData is the uninstaller's checkbox. It removes only %APPDATA%\com.helioxos.app and
; %LOCALAPPDATA%\com.helioxos.app (the web view's storage and cache); Tauri's own label, "Delete the application
; data", can be read as the person's settings and data, which it never touches.
; helioxLeftBehind is Heliox's own: the message at the end of an interactive uninstall (installer-hooks.nsh).

LangString addOrReinstall ${LANG_ENGLISH} "Add or reinstall components"
LangString alreadyInstalled ${LANG_ENGLISH} "Already installed"
LangString alreadyInstalledLong ${LANG_ENGLISH} "${PRODUCTNAME} ${VERSION} is already installed. Choose what to do, then click Next."
LangString appRunning ${LANG_ENGLISH} "${PRODUCTNAME} is open. Close it, then try again."
LangString noPermissionToCloseApp ${LANG_ENGLISH} "${PRODUCTNAME} is open and this installer is not allowed to close it. Close it, then try again."
LangString appRunningOkKill ${LANG_ENGLISH} "${PRODUCTNAME} is open.$\nClick OK to close it."
LangString chooseMaintenanceOption ${LANG_ENGLISH} "Choose what to do with this installation."
LangString chooseHowToInstall ${LANG_ENGLISH} "Choose how to install ${PRODUCTNAME}."
LangString createDesktop ${LANG_ENGLISH} "Create a desktop shortcut"
LangString dontUninstall ${LANG_ENGLISH} "Do not uninstall"
LangString dontUninstallDowngrade ${LANG_ENGLISH} "Do not uninstall (this installer cannot install an older version over a newer one)"
LangString failedToKillApp ${LANG_ENGLISH} "${PRODUCTNAME} could not be closed. Close it, then try again."
LangString installingWebview2 ${LANG_ENGLISH} "Installing WebView2..."
LangString newerVersionInstalled ${LANG_ENGLISH} "A newer version of ${PRODUCTNAME} is already installed. Installing an older one is not recommended. To install it anyway, uninstall the current version first. Choose what to do, then click Next."
LangString older ${LANG_ENGLISH} "older"
LangString olderOrUnknownVersionInstalled ${LANG_ENGLISH} "An $R4 version of ${PRODUCTNAME} is installed. Uninstalling it before installing this one is recommended. Choose what to do, then click Next."
LangString silentDowngrades ${LANG_ENGLISH} "A silent install cannot replace a newer version with an older one.$\nUninstall the newer version first, or run this installer without /S."
LangString unableToUninstall ${LANG_ENGLISH} "The installed version could not be uninstalled."
LangString uninstallApp ${LANG_ENGLISH} "Uninstall ${PRODUCTNAME}"
LangString uninstallBeforeInstalling ${LANG_ENGLISH} "Uninstall before installing"
LangString unknown ${LANG_ENGLISH} "unknown"
LangString webview2AbortError ${LANG_ENGLISH} "WebView2 could not be installed. The app cannot run without it. Run the installer again."
LangString webview2DownloadError ${LANG_ENGLISH} "The WebView2 installer could not be downloaded: $0"
LangString webview2DownloadSuccess ${LANG_ENGLISH} "WebView2 installer downloaded"
LangString webview2Downloading ${LANG_ENGLISH} "Downloading the WebView2 installer..."
LangString webview2InstallError ${LANG_ENGLISH} "WebView2 could not be installed (exit code $1)"
LangString webview2InstallSuccess ${LANG_ENGLISH} "WebView2 installed"
LangString deleteAppData ${LANG_ENGLISH} "Also delete the app's web view data (your Heliox settings and data are kept)"
LangString helioxLeftBehind ${LANG_ENGLISH} "${PRODUCTNAME} is uninstalled. Nothing of yours was deleted.$\n$\nProgram files are still in $PROFILE\.config\heliox-os: env, python, tools, uv-cache and setup (several hundred MB). Delete those folders to remove them.$\n$\nWorking files (logs, web view data) are safe to delete: $PROFILE\.local\state\heliox-os, $LOCALAPPDATA\heliox-os, $LOCALAPPDATA\com.helioxos.app.$\n$\nYour settings, memory and receipts are kept, in $PROFILE\.config\heliox-os and $PROFILE\.local\share\heliox-os."
