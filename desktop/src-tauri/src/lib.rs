//! Copytrader desktop shell.
//!
//! A deliberately thin Tauri host: it loads the statically-exported Next.js frontend
//! (`frontend/out`) into a native webview. All data flows over plain browser `fetch` to the
//! backend URL the user sets in Settings, so there are no custom Tauri commands to maintain —
//! repointing at the Windows box or the Beelink is a runtime setting, not a rebuild.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
