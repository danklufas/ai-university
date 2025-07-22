# Quick Reminder Cheat Sheets

## 1. Everyday Workflow (Edit → Preview → Commit → Deploy)

**Local preview**
```bash
python -m mkdocs serve
Open: http://127.0.0.1:8000/
(Leave terminal running; Ctrl+C stops it.)

Commit & push

bash
Copy
Edit
git add docs/...
git commit -m "Your message"
git push origin main
Publish to GitHub Pages

bash
Copy
Edit
python -m mkdocs gh-deploy
2. Common Git Commands
bash
Copy
Edit
git status                 # see what changed
git add <file>             # stage a file
git add .                  # stage everything
git commit -m "message"    # commit staged changes
git push origin main       # push to GitHub
git pull origin main       # pull latest from GitHub
Undo a staged file:

bash
Copy
Edit
git restore --staged <file>
3. MkDocs Basics
Edit pages in docs/

Navigation lives in mkdocs.yml under nav:

Rebuild local site: python -m mkdocs serve

Deploy: python -m mkdocs gh-deploy

Add a new page
Create docs/new-page.md

Add to mkdocs.yml:

yaml
Copy
Edit
nav:
  - Home: index.md
  - Week 1: week-1.md
  - Week 2: week-2.md
  - Cheat Sheets: cheat-sheets.md
  - New Page: new-page.md
4. VS Code Tips
Toggle terminal: Ctrl + `

Command Palette: Ctrl + Shift + P

Find/Replace regex: Ctrl + H, click .*

Reload the window (fix UI glitches): Developer: Reload Window

5. Troubleshooting
Local site won’t load

Make sure python -m mkdocs serve is running

Use http://127.0.0.1:8000/ (colon, not dot)

GitHub Pages not updated

Did you run python -m mkdocs gh-deploy after pushing?

Wait ~30 seconds and refresh

site/ shows up in Git changes

Ensure .gitignore contains site/

If already tracked:

bash
Copy
Edit
git rm -r --cached site
git commit -m "Remove generated site"
git push origin main
(Add anything else you want to remember here as you go.)

yaml
Copy
Edit

5. Save the file (**Ctrl+S**).

---

Reply **“done”** after you’ve saved `cheat-sheets.md`.  
Then we’ll do **Step 2: add it to the nav and push/deploy**.








