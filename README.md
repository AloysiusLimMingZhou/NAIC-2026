# Ensemble Model Run Guide
### ConvNeXt-Small (Exp 7) + EfficientNet-B3 (Exp 5) → Break 80% Accuracy

**What to do:** Start the cloud server → access Jupyter → configure paths → run all 5 strategies → record and push results.  
**Estimated time:** ~30–45 minutes (most of it is GPU inference, not manual work).

---

## Part 1 — Set Up the Git Repository

All of this is done inside a **Terminal** in Windows.

### Step 1.1 — Open a Terminal
Do `ctrl + r` to and enter powershell to open a powershell in Windows.

### Step 1.2 — Navigate to the Project Folder

> **Clone the repo first**
> ```bash
> git clone https://github.com/AloysiusLimMingZhou/NAIC-2026.git
> cd NAIC-2026
> ```

### Step 1.3 — Set Your Git Identity (first time only)
If you've never committed from this server before:
```bash
git config user.name "Your Name"
git config user.email "your-email@example.com"
```

### Step 1.4 — Fetch All Latest Branches
```bash
git fetch --all
```

### Step 1.5 — Switch to the Ensemble Feature Branch
```bash
git checkout features/ensemble_model
```

Expected output:
```
Switched to branch 'features/ensemble_model'
Your branch is up to date with 'origin/features/ensemble_model'.
```

> **If you see `error: pathspec 'features/ensemble_model' did not match`:**
> ```bash
> git checkout -b features/ensemble_model origin/features/ensemble_model
> ```

### Step 1.6 — Pull the Latest Changes
```bash
git pull origin features/ensemble_model
```

This downloads anything your teammates have already pushed to this branch.

### Step 1.7 — Confirm Your Current Branch
```bash
git branch
```

The output should show `* features/ensemble_model` with an asterisk next to it. If it doesn't, something went wrong — go back to Step 1.5.

---

## Part 2 — Start the Cloud Server

### Step 2.1 — Go to Discord
Open Discord and navigate to the **#general** or **#notification** channel.

### Step 2.2 — Start the VM
Type the following command and press **Enter**:

```
/VM-start
```

Wait for the bot to reply. It will send a message confirming the VM is starting up. This usually takes **1 minute**.

> **If the bot respond with an error message:** This usually means that the GCP server is busy and we can't access it. So just wait for 15-30 minutes and try restart. If its still erroring, tag someone to assist you.

### Step 2.3 — Wait for Confirmation
The bot will send a message like:
```
✅ VM started! Your Jupyter server is ready
```

Then, you may go to this Jupyter Server URL: https://jupyter.chocorot.net/

---

## Part 3 — Log Into the Jupyter Server

### Step 3.1 — Open the URL in Your Browser
Paste the URL into your browser's address bar and press Enter.

You'll see the **TLJH (The Littlest JupyterHub) login page**.

### Step 3.2 — Log In
Enter your credentials:

| Field | Value |
|---|---|
| **Username** | *(the name your teammate assigned you)* |
| **Password** | *(the password your teammate assigned you)* |

Click **Sign in**.

> **If login fails:** Make sure there are no spaces before/after your username. Passwords are case-sensitive.

### Step 3.3 — You're In
You should now see the **JupyterLab file browser** on the left. You're in your home directory on the cloud server.

---

## Part 4 — Upload the Notebook to the Server

You need to upload `ensemble_ultimate_fixed.ipynb` from your **local machine** to the **correct folder** on the server.

### Step 4.1 — Upload the File
1. Click the **Upload** button — it looks like an **up arrow icon (↑)** in the file browser toolbar at the top.
2. A file picker dialog opens on your computer.
3. Find and select `ensemble_ultimate_fixed.ipynb` on your local machine.
4. Click **Open**.

The file will appear in the `naic-project/` folder in the sidebar.

---

## Part 5 — Run the Notebook

### Step 5.1 — Run All Cells
In the JupyterLab menu bar, click:

```
Kernel → Restart Kernel and Run All Cells...
```

A dialog box will appear asking **"Are you sure you want to restart the kernel and run all cells?"** — click **Restart**.

> **Why restart first?** This clears any leftover variables from previous runs. Always restart before a full run to avoid stale state bugs.

### Step 6.2 — Watch the Progress
Cells run top to bottom. Each running cell shows `[*]` in the top-left corner. A completed cell shows a number like `[1]`, `[2]`, etc.

Key cells and what to expect:

| Cell | What it does | How long |
|---|---|---|
| Cell 1 | Installs pip packages | 1–3 min (skip if already installed) |
| Cell 3 | Downloads dataset | Skip if images already on disk |
| Cell 10 | Validates your flags | Instant — errors here mean bad config |
| Cell 11 | Downloads HuggingFace processors | 30 sec |
| Cell 13 | **Main inference** — loads all 10 fold models + runs 8-view TTA | **15–25 min** |
| Cell 15 | Runs all 5 strategies | < 1 min |
| Cell 16 | Prints comparison table + best strategy | Instant |
| Cell 17 | Saves results to `ensemble_results/` | Instant |

### Step 5.3 — Wait for Completion
The full run takes **20–35 minutes** depending on GPU availability. The notebook is done when Cell 17 finishes and prints:

```
Results saved to: ensemble_results/ensemble_<timestamp>.json
```

## Part 6 — Save and Push Results to Git

### Step 6.0 — Verify the best strategy
- Inside your Jupyter Server, every time you've run the notebook, the result metrics will be saved in json files. 
- Make sure every time you've run an experiment, create a folder and named it as exp_1_results, exp_2_results, ... and save the 'ensemble_results/ensemble_{timestamp}.json' in the corresponding folder.

### Step 6.1 — Download all the experiment metrics
In each experiment, find 'ensemble_results/ensemble_{timestamp}.json' and right click the file, find the Download button (down arrow icon) in the JupyterLab toolbar to download the files to your local machine.

### Step 6.2 — Navigate to the Project Folder in your VS Code
- Once you've downloaded the files, open VS Code and navigate to the project folder. 
- Then, move the downloaded metrics json files to the 'ensemble_results' folder in your local machine. 
- Then, open a new terminal.

### Step 6.3 — Check What Has Changed
```bash
git status
```

You should see output like:
```
On branch features/ensemble_model
Changes not staged for commit:
  modified:   ensemble_ultimate_fixed.ipynb

Untracked files:
  ensemble_results/ensemble_20260424_143022.json
```

### Step 6.4 — Stage All Changes
```bash
git add .
```

> **Tip:** Run `git status` again after staging to confirm all files are listed under "Changes to be committed" (shown in green).

### Step 6.5 — Commit with a Descriptive Message
Use this format so your teammates can read the history clearly:

```bash
git commit -m "feat(ensemble): run all 5 strategies, best=per_class_weighted 80.35% acc"
```

Replace the numbers with your actual results. Follow this format:
```
feat(ensemble): run all 5 strategies, best=<strategy_name> <accuracy>% acc
```

### Step 6.6 — Push to the Feature Branch
```bash
git push origin features/ensemble_model
```

Expected output:
```
Enumerating objects: 7, done.
Counting objects: 100% (7/7), done.
Delta compression using up to 8 threads
Compressing objects: 100% (5/5), done.
Writing objects: 100% (5/5), 1.23 MiB | 4.56 MiB/s, done.
Total 5 (delta 2), reused 0 (delta 0), pack-reused 0
To https://github.com/<your-org>/<your-repo>.git
   a1b2c3d..e4f5g6h  features/ensemble_model -> features/ensemble_model
```

> **If push is rejected with `non-fast-forward` error:** Someone else pushed to the branch while you were working. Run:
> ```bash
> git pull origin features/ensemble_model --rebase
> git push origin features/ensemble_model
> ```

### Step 7.6 — Confirm the Push
```bash
git log --oneline -5
```

Your commit should appear at the top of the list.

---

## Quick Reference — All Git Commands in Order

```bash
# ── Setup (once per server, skip if already done) ─────────────────
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
git clone https://github.com/<org>/<repo>.git naic-project

# ── Every session ─────────────────────────────────────────────────
cd ~/naic-project
git fetch --all
git checkout features/ensemble_model
git pull origin features/ensemble_model

# ── After running the notebook ────────────────────────────────────
git status
git add .
git commit -m "feat(ensemble): run all 5 strategies, best=<strategy> <acc>% acc"
git push origin features/ensemble_model

# ── Verify push ───────────────────────────────────────────────────
git log --oneline -5
```

---