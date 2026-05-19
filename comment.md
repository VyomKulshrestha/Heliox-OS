@VyomKulshrestha Created a clean PR (#239) with all security fixes applied:
- Replaced eval() with json.loads() to prevent RCE
- Added graceful fallback for None executor
- Based on latest main

PR: https://github.com/VyomKulshrestha/Heliox-OS/pull/239