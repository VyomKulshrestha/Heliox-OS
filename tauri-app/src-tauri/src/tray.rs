use tauri::{
    App, Manager,
    menu::{IsMenuItem, Menu, MenuItem},
    tray::TrayIconBuilder,
};

pub fn setup_tray(app: &App) -> Result<(), Box<dyn std::error::Error>> {
    let show = MenuItem::with_id(app, "show", "Show Pilot", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    
    // FIX 1: Rust requires explicit coercion to trait objects (`&dyn IsMenuItem<_>`)
    // for the slice elements passed to `with_items`.
    let menu = Menu::with_items(app, &[
        &show as &dyn IsMenuItem<_>,
        &quit as &dyn IsMenuItem<_>
    ])?;

    let mut tray_builder = TrayIconBuilder::new()
        .menu(&menu)
        .tooltip("Pilot — AI Command Center")
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
            "quit" => {
                app.exit(0);
            }
            _ => {}
        });

    // FIX 2: A tray icon typically requires an icon image to build and display successfully across OSs.
    // We securely clone the app's default window icon if it exists.
    if let Some(icon) = app.default_window_icon() {
        tray_builder = tray_builder.icon(icon.clone());
    }

    let _tray = tray_builder.build(app)?;

    Ok(())
}