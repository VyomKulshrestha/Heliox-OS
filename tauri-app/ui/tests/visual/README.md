# Visual Regression Tests

Playwright-based visual regression suite for the Heliox OS UI.  
Tests run against the **Vite dev server** (`http://localhost:1420`) — no Tauri binary or daemon required.

## Structure

```
tests/visual/
├── helpers.ts              # Shared setup: Tauri IPC mock, navigation, animation freeze
├── chat.spec.ts            # Chat interface (empty state, input bar, messages)
├── settings.spec.ts        # Settings panel (all sections, toggles, light mode)
├── agent-thoughts.spec.ts  # ReActPipeline (idle, skeleton, active, completed, thoughts)
├── __snapshots__/          # Committed baseline PNG screenshots
└── README.md               # This file
```

## Running locally

```bash
cd tauri-app/ui

# Install Playwright (first time only)
npx playwright install --with-deps chromium

# Run all visual tests
npm run test:visual

# Update baselines after intentional UI changes
npm run test:visual:update
```

## How it works

1. Each test navigates to the app with the SetupWizard skipped via `localStorage`
2. A Tauri IPC stub prevents crashes when the app calls `window.__TAURI_INTERNALS__`
3. CSS animations are frozen for pixel-stable screenshots
4. `toHaveScreenshot()` diffs the current render against the committed baseline PNG
5. If `maxDiffPixelRatio` (0.2%) is exceeded the test fails and a diff image is saved to `test-results/`

## Updating baselines

Run this after **intentional** UI changes (new feature, design update):

```bash
npm run test:visual:update
```

Then commit the updated PNGs in `__snapshots__/` alongside your code changes.

## CI behaviour

The `visual-regression` job in `.github/workflows/ci.yml`:
- Runs on every PR against `main`
- Uploads diff images as artifacts (retained 7 days) when tests fail
- Reviewers can download the artifact to see exactly what changed visually

## Pixel diff tolerance

| Setting | Value | Reason |
|---|---|---|
| `maxDiffPixelRatio` | `0.002` (0.2%) | Allows minor font/anti-aliasing differences across OS |
| `threshold` | `0.1` | Per-pixel colour distance (0–1 scale) |
| `animations` | `disabled` | Freezes CSS transitions for stable snapshots |
