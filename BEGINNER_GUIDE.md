# Bank Subscription Predictor: a beginner's guide

This guide is for someone who has never run a coding project before. The commands below are for **Windows PowerShell**.

You can do three separate things:

1. **Open the finished website on your computer.** This is the easiest starting point. No model training is required.
2. **Run the Python scripts.** This lets you make predictions, rebuild the trained model, and generate the notebook.
3. **Publish the website using GitHub and Vercel.** Other people can then visit it using a normal web link.

## Before you start: get the right project files

Ask the project owner for the latest project folder or ZIP file. If it is a ZIP, right-click it and choose **Extract All** before continuing.

**Version note:** use the latest repository files so the website and trained model match.

Public repository: https://github.com/jarrodfranktamargocics-ui/DS_4_DecisionTreeClassifier_byte

You can get a copy from GitHub using **Code → Download ZIP**. Extract the ZIP.

Open the extracted folder. You should see files such as `README.md`, `train.py`, `requirements.txt`, and `vercel.json`, plus a folder called `web`. This is the **project folder**, also called the **repository root**. If you only see another folder, open that inner folder.

Do not copy someone else's `.venv` folder to use on your computer. Create your own Python environment in Part 2.

## A few words explained

| Word | What it means here |
|---|---|
| Terminal / PowerShell | A window where you type commands for your computer. |
| Script | A file containing instructions, such as `train.py`. |
| Model | The decision tree that has learned patterns from historical data. |
| Training | The process of learning those patterns. |
| Local website | A website running on your own computer. |
| GitHub repository | An online project folder with a history of changes. |
| Commit | A saved set of changes in that history. |
| Push | Upload your saved changes to GitHub. |
| Deploy | Publish the website to a hosting service, here Vercel. |

## Part 1 — Open the website on your computer

### Step 1: install Python

Use **Python 3.12** for this project's training setup. Get it from the [official Python website](https://www.python.org/downloads/windows/). Python runs the scripts and can also serve the website locally.

After installing, close any old terminal windows and open a new one. Verify that the Python launcher can find Python 3.12:

```powershell
py -3.12 --version
```

You should see `Python 3.12` followed by a patch number. If the command fails, see Troubleshooting below.

### Step 2: open PowerShell inside the project folder

1. Open the project folder in File Explorer.
2. Click the address bar at the top of File Explorer.
3. Type `powershell` and press **Enter**.
4. A terminal opens in that folder.

Check that you are in the correct place:

```powershell
dir
```

You should see `train.py`, `requirements.txt`, and `web` in the list. Copy only the commands inside the boxes in this guide, not the box labels or your terminal's `PS C:\...>` prompt.

### Step 3: start the local website

```powershell
py -3.12 -m http.server 8000 --bind 127.0.0.1 --directory web
```

Keep this terminal open. A message containing `Serving HTTP` means it is running. It is normal for the terminal to remain busy and print lines when you browse.

### Step 4: open the website

Open Chrome, Edge, or another browser. Type this into the browser's address bar:

```text
http://127.0.0.1:8000/
```

You should see **Bank Subscription Predictor** and navigation for Predictor, Performance, Model, and About.

Try it:

1. Choose a sample profile, or change some customer fields.
2. Press **Explain prediction** after editing the fields.
3. Read the predicted decision, model score, and explanation.
4. Visit **Performance** to compare the original and tuned models.
5. Visit **Model** to learn about feature importance and the training process.

The score is a model score, not a guaranteed probability of a customer subscribing. The explanation describes the model's reasoning.

**You do not need to train the model or install the machine-learning libraries just to open this website.** The browser uses the saved `web/model.json` file.

Do not double-click `web/index.html` to run the demo. Serve it using the command above so the browser can load the model data correctly.

### Stop it or start it again

To stop the website, click the terminal and press **Ctrl+C**. To start it later, open PowerShell in the same project folder and run the server command again.

The local address works only on your computer with this setup. Sending `127.0.0.1` to a friend will not let them see your website. Part 3 creates a shareable link.

## Part 2 — Run the model and training scripts

This part is optional if you only want to view or publish the existing website. Use a second PowerShell window if the first one is running the web server.

### Step 1: create an isolated Python environment

Run this from the project folder, once:

```powershell
py -3.12 -m venv .venv
```

This creates a folder named `.venv`. Think of it as a toolbox belonging only to this project.

We will call its Python program directly, so **you do not need to activate the environment or change PowerShell's security settings**.

### Step 2: install the required libraries

Run these commands one at a time:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This downloads the tools listed in `requirements.txt`. Internet access is required. Wait for the command to finish and for the PowerShell prompt to return. Installation is normally needed only once per environment.

### Step 3: make one prediction using the saved model

If your copy includes `outputs/trained_pipeline.joblib`, run:

```powershell
.\.venv\Scripts\python.exe predict.py --example
```

You should see a result containing `prediction`, `score`, `threshold`, `baseline`, and `shap`.

- `prediction`: yes or no.
- `score`: the model's numeric score, between 0 and 1.
- `threshold`: the cutoff used to turn that score into yes or no.
- `shap`: the contributions of the features to this prediction.

If the saved model file is missing, complete Step 4 first. Only use trusted `.joblib` files supplied by the project owner or generated by your own training run.

### Step 4: train the model from the beginning

**This is not necessary every time you open the website.** Training downloads the public dataset if needed, compares 55 candidate configurations using training-validation folds, selects a model and cutoff, and writes new model/evaluation files.

It also **overwrites generated outputs**, including the browser model and metrics. Make a copy of the project folder first if you want to keep the existing results unchanged.

```powershell
.\.venv\Scripts\python.exe -u train.py
```

Keep the computer awake and let it finish. It can take several minutes or longer depending on your computer. Progress messages will show candidate numbers, and the final output will include the model metrics.

The main generated files are:

| File | What it contains |
|---|---|
| `outputs/trained_pipeline.joblib` | The trained Python model, preprocessing, and cutoff. |
| `web/model.json` | The trained tree and supporting data used by the website. |
| `outputs/metrics.json` | Accuracy, precision, recall, F1, and other results. |
| `outputs/confusion_matrix.png` | A picture comparing predicted and actual outcomes. |
| `outputs/feature_importance.png` | A picture of the most important features. |
| `outputs/model_summary.md` | A written model summary. |

The dataset is automatically stored in `data/bank-full.csv`. Do not edit it to try to improve the scores.

After training, refresh the website to load the newly exported model. Higher scores are not guaranteed simply because training ran again.

### Step 5: generate the executed notebook

After training has finished:

```powershell
.\.venv\Scripts\python.exe make_notebook.py
```

This creates and runs `notebooks/bank_marketing.ipynb`. A notebook combines explanations, code, and results in one document. The script reuses the saved model selection and recomputes the evaluation; it also refreshes generated outputs. It does not require you to open a notebook editor first.

Once uploaded, the notebook can be viewed on GitHub. You do not need to install Jupyter just to generate it with this script.

### Step 6: check that browser predictions match Python

For this optional check, install Node.js from [nodejs.org](https://nodejs.org/), then open a new terminal in the project folder and run:

```powershell
node tests/model.test.mjs
```

Expect a message beginning with `Passed:`. The current saved model has 157 prediction/SHAP cases. That count may change after retraining. You do not need `npm install` to run this test.

### Other scripts: you can skip these initially

| Command | Purpose |
|---|---|
| `.\.venv\Scripts\python.exe -u experiment.py` | Repeat the full model-selection search. `train.py` already runs this, so do not run both unless you intend to repeat the work. |
| `.\.venv\Scripts\python.exe -u explore_targets.py` | Repeat the additional 12-configuration accuracy/recall study. This saves research results without replacing the website's model. Run the main training workflow first if its prerequisite results are missing. |

## Part 3 — Put the website online with GitHub and Vercel

Python trains the model on your computer. **Vercel hosts the finished `web` folder; it does not train the model or run these Python scripts.** Predictions and explanations run in each visitor's browser.

### Step 1: upload the latest project to GitHub

You need a GitHub account and a repository you own or have permission to update. A friend should use their own repository for their own deployment.

For a new repository, the browser method is:

1. Sign in at [github.com](https://github.com/).
2. Create a new repository. For this internship submission, use `DS_4_DecisionTreeClassifier_byte` under the appropriate owner's account and make it public.
3. Use **Add file → Upload files**, or the upload link offered for an empty repository.
4. Upload the project files and folders, preserving their structure. The `web` folder and `vercel.json` must sit directly at the repository's top level.
5. Enter a message such as `Add bank subscription predictor` and choose **Commit changes**.

Do **not** upload a ZIP as the website: GitHub/Vercel need the extracted files. Do not accidentally put the entire project inside an extra outer folder.

Include source scripts, `requirements.txt`, `package.json`, `vercel.json`, `.gitignore`, documentation, `web`, `tests`, `notebooks`, and `outputs`. The internship repository should contain the notebook and evaluation deliverables as well as the website.

Leave out `.venv`, `.git`, `.runtime`, `__pycache__`, `node_modules`, `.vercel`, downloaded `data`, and any `.env` files or credentials. A manual browser upload does not automatically apply `.gitignore`. If an upload is rejected because it is too large, use [GitHub Desktop](https://desktop.github.com/) instead of removing required deliverables.

For an **existing cloned repository**, GitHub Desktop is a convenient alternative: add/open that local repository, review the changed files, write a summary, **Commit**, and **Push origin**. Commit saves locally; Push uploads to GitHub. Downloading a ZIP does not create a cloned repository, so do not expect it to work identically in Desktop.

Before proceeding, view the repository on GitHub and verify these files exist:

```text
vercel.json
web/index.html
web/performance.html
web/insights.html
web/about.html
web/model.json
web/baseline.json
```

Also include all the CSS, JavaScript, and image files inside `web`.

### Step 2: connect Vercel to the repository

1. Visit [vercel.com](https://vercel.com/) and sign in using your own account.
2. Choose **Add New → Project** (the wording may be **New Project**).
3. Connect GitHub if prompted. Give Vercel access to the repository you intend to deploy.
4. Find that repository and choose **Import**.

Git repository import is described in [Vercel's official Git documentation](https://vercel.com/docs/git).

### Step 3: check the deployment settings

The included `vercel.json` already defines the static-site settings. Keep the project rooted at the repository top level:

| Setting | Value for this project |
|---|---|
| Framework preset | **Other** |
| Root directory | Repository root; leave at the default top level |
| Output directory | `web` |
| Build command | Empty — no build step |
| Install command | Empty — no dependency installation step |
| Environment variables | None required |

If Vercel shows inferred build/install commands, ensure the existing configuration leaves them empty. Do not enter the word `empty` as a command. Do not set both Root Directory and Output Directory to `web`; that would point to the wrong nested location.

These settings match [Vercel's static-site build configuration guidance](https://vercel.com/docs/builds/configure-a-build).

### Step 4: deploy and test the live site

1. Click **Deploy**.
2. Wait for the deployment to finish successfully.
3. Open the provided website URL, usually ending in `.vercel.app`.
4. Test the Predictor, Performance, Model, and About pages.
5. Edit a profile and press **Explain prediction**.
6. Open the link on your phone or in a private browser window to check that visitors can access it.

If visitors see a Vercel login requirement, check that you shared the intended production URL and review the project's deployment-protection settings before changing access.

You can now share the live URL. Your computer and local Python server do not need to stay running.

### Step 5: publish future changes

1. Edit the project locally and test it.
2. If the model changed, complete training/export and the model checks first.
3. Upload the changed files to GitHub, or commit and push them.
4. Vercel normally creates a new deployment from connected Git updates. Updates to the configured production branch update the live production website; other branches can produce previews.
5. Check that the latest deployment succeeded, then open the live URL and verify the changes.

This automatic deployment behavior is documented in [Vercel's deployment guide](https://vercel.com/docs/deployments). Merely saving a file on your laptop does not update GitHub or the live site.

## Troubleshooting

| What you see | What to do |
|---|---|
| `py` is not recognized | Install Python with the Windows launcher, then open a new terminal. If `python --version` already shows 3.12, you can use `python` instead of `py -3.12`. |
| Python 3.12 cannot be found | Install that version, or ask the owner to help select the installed interpreter. |
| `requirements.txt` or `train.py` cannot be found | You opened PowerShell in the wrong folder. Return to the folder containing those files. |
| `.venv\Scripts\python.exe` cannot be found | Create the environment using Part 2, Step 1, and confirm you are in the project folder. |
| A Python library is missing | Install `requirements.txt` using the `.venv` Python command shown above, then run scripts with that same Python. |
| A package fails to install | Check the Python version and internet connection. Keep the error message and send it to the owner; do not randomly change the pinned package versions. |
| The browser says the local site cannot be reached | Start the server and keep its terminal open. Check the address and port number. |
| Port 8000 is already in use | Use `py -3.12 -m http.server 8001 --bind 127.0.0.1 --directory web`, then visit `http://127.0.0.1:8001/`. |
| The website says the model could not load | Use the local server rather than opening an HTML file directly. Confirm `web/model.json` exists. |
| Editing a field does not immediately change the result | Press **Explain prediction**. Check for an empty or invalid required field. |
| Python prediction says a model file is missing | Run training, or get the trusted saved model from the owner. |
| The site still looks old | Confirm you have the latest files and refresh with **Ctrl+F5**. For Vercel, also check the latest Git commit and deployment status. |
| Vercel shows a 404 | Check that Root Directory is the repository root, Output Directory is `web`, and `web/index.html` was uploaded. |
| Vercel cannot find your GitHub repository | Check which GitHub account is connected and whether Vercel has access to that repository. |

## Quick checklist

- To view locally: get the latest files, open PowerShell in the project folder, run the HTTP server, and open the local URL.
- To train: create `.venv`, install requirements, run `train.py`, generate the notebook, and run the model checks.
- To go live: upload the latest extracted project to GitHub, import it into Vercel, check the settings, and deploy.
- To update the live site: test locally, commit/upload the changes to GitHub, and verify Vercel's new deployment.

These instructions explain how to run your own copy and deploy it. A local address is never the public live URL.
