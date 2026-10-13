# Push this project to GitHub

## 1. Create an empty GitHub repository

On GitHub, select **New repository**. Use a name such as `cyberbullying-detection-prototype`. Do not add a README, license, or `.gitignore` on GitHub because this project already has those files.

## 2. Open a terminal in this `Research_Prototype` folder

Check the current branch:

```bash
git branch --show-current
```

The project history contains versions V1.0 through V2.0. To connect your new GitHub repository, replace `YOUR-USERNAME` with your GitHub username:

```bash
git remote add origin https://github.com/YOUR-USERNAME/cyberbullying-detection-prototype.git
git push -u origin master
```

If GitHub shows a different default branch preference, the local branch can be renamed before pushing:

```bash
git branch -M main
git push -u origin main
```

Use one branch approach, not both. GitHub may ask you to sign in or authorize Git Credential Manager.

## 3. If you want to start a brand-new Git history instead

Only do this if you intentionally do not want to preserve the existing version history. Remove the `.git` folder, then run `git init`, `git add .`, `git commit -m "Initial research prototype"`, add your remote, and push.

## 4. Reproduce the experiment before pushing (recommended)

```bash
pip install -r requirements.txt
python src/train_model.py
pytest -q
python src/predict.py "sample social media text"
```

The trained model files are generated under `results/model/` and are excluded from Git. This keeps the repository reproducible without committing binary model files.

## Important

Before making the repository public, confirm that you are allowed to redistribute the included dataset. If not, keep the repository private or provide instructions for obtaining the data lawfully instead of publishing the CSV.
