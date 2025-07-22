Quick Reminder Cheat Sheets
## 1. Everyday Workflow (Edit → Preview → Commit → Deploy)
**Local preview**

```bash
python -m mkdocs serve
```
Open: <http://127.0.0.1:8000/>
(Leave terminal running; **Ctrl+C** stops it.)

**Commit & push**
```bash
git add docs/...
git commit -m "Your message"
git push origin main
```
**Publish to GitHub Pages**
```bash
python -m mkdocs gh-deploy
```

## 2. Common Git Commands
```bash
git status                 # see what changed
git add <file>             # stage a file
git add .                  # stage everything
git commit -m "message"    # commit staged changes
git push origin main       # push to GitHub
git pull origin main       # pull latest from GitHub
```

Undo a staged file:
```bash
git restore --staged <file>
```

## 3. MkDocs Basics
  
Edit pages in `docs/`
Navigation lives in `mkdocs.yml` under `nav:`

**Rebuild local site**
```bash
python -m mkdocs serve
```

**Deploy to GitHub Pages**
```bash
python -m mkdocs gh-deploy
```

### Add a new page
1. Create `docs/new-page.md`
2. Add to `mkdocs.yml`:
```yaml
nav:
  - Home: index.md
  - Week 1: week-1.md
  - Week 2: week-2.md
  - Cheat Sheets: cheat-sheet.md
  - New Page: new-page.md
  ```

## 4. VS Code Tips
Toggle terminal: **Ctrl + `**
Command Palette: **Ctrl + Shift + P**
Find/Replace regex: **Ctrl + H**, click `.*`
Reload the window (fix UI glitches): **Developer: Reload Window**

  ## 5. Troubleshooting
```bash
python -m mkdocs serve
```

Use `http://127.0.0.1:8000/` (colon, not dot).

### GitHub Pages not updated
```bash
python -m mkdocs gh-deploy
```

Wait ~30 seconds and refresh the live site.

### `site/` shows up in Git changes
Ensure `.gitignore` contains `site/`.

If it's already tracked:
```bash
git rm -r --cached site
git commit -m "Remove generated site"
git push origin main
```