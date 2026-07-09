Hi, thanks for working on this! The Rust implementation for extracting PDF text natively is a great architectural choice, and bumping the max file size makes sense. 

However, there is a bug with the drag-and-drop event handlers: you have implemented **both** the native Tauri webview drag-and-drop listener (`getCurrentWebview().onDragDropEvent`) **and** the HTML5 DOM listener (`<svelte:window ondrop={handleDrop} />`). 

Because both listeners are active, dropping a file triggers both of them simultaneously. They both independently call `extract_file_text` and append to the `attachments` array, which means dragging a single file causes it to appear **twice** in the UI. 

Could you please remove the `<svelte:window>` HTML5 drag handlers entirely and rely solely on the Tauri native API you wrote? Once that's fixed, we can get this merged!
