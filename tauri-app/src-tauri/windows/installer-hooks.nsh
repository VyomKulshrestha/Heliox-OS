; Heliox's hooks in the Windows installer (bundle.windows.nsis.installerHooks in tauri.conf.json).
;
; Before uninstalling: end the app and the daemon its environment runs (end-heliox.ps1). The installer's own check
; ends only the app, abruptly, which left the daemon running from the person's environment.
; After uninstalling, in an interactive uninstall: say what is left in the home folder and that nothing of the
; person's was deleted. A silent uninstall (/S) shows nothing, so it says nothing.
; Neither runs during an update (/UPDATE), when the next version's installer runs this uninstaller.

; The folder of this file, fixed now: inside a macro ${__FILEDIR__} would name the installer's own script.
!define HELIOX_HOOKS_DIR "${__FILEDIR__}"

!macro NSIS_HOOK_PREUNINSTALL
  ${If} $UpdateMode <> 1
    InitPluginsDir
    File "/oname=$PLUGINSDIR\end-heliox.ps1" "${HELIOX_HOOKS_DIR}\end-heliox.ps1"
    System::Call 'Kernel32::SetEnvironmentVariable(t "HELIOX_APP", t "$INSTDIR\${MAINBINARYNAME}.exe")'
    nsExec::ExecToLog 'powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$PLUGINSDIR\end-heliox.ps1"'
    Pop $0
  ${EndIf}
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  ${If} $UpdateMode <> 1
  ${AndIfNot} ${Silent}
    MessageBox MB_OK|MB_ICONINFORMATION "$(helioxLeftBehind)"
  ${EndIf}
!macroend
