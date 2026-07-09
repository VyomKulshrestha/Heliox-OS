Hi, thanks for adding the Forensics Runbook documentation! It is very well written and accurately explains the severity matrix and containment pipeline.

However, your local branch seems to be severely out of sync with the `main` branch. Because you modified an older version of `README.md`, this PR unintentionally deletes the documentation for the newly added `Calendar Agent` and completely deletes the PyTorch installation FAQ (Q8).

Could you please run `git pull origin main` to fetch the latest changes, and then rebase or merge them into your branch to resolve the conflicts? Once the `README.md` is synced up so it doesn't overwrite other contributors' work, we can get this merged!
