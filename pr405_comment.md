Hi, thanks for working on the Semantic Search Agent! The integration with the `@auto_register` decorator and the new `WORKSPACE_INDEX` / `WORKSPACE_SEARCH` action types looks very clean.

However, it looks like you forgot to commit the actual `workspace_index.py` file! The `SemanticSearchAgent` explicitly imports `from pilot.memory.workspace_index import WorkspaceIndex` and calls its methods, but that file isn't included in the pull request diff. 

If we merge this right now, the daemon will instantly crash on startup with a `ModuleNotFoundError: No module named 'pilot.memory.workspace_index'`. Could you please commit and push that file, along with any necessary vector database dependencies (e.g. Chroma, FAISS) to `pyproject.toml`? Thanks!
