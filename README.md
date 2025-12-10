# SADHI

A loss function designed for Image Quality Assessment (IQA) problems, balancing image fidelity, naturalness, and semantic context to optimize perceived image quality.

## Structure

- `configs/`: Configuration files (YAML).
- `data/`: Datasets.
- `notebooks/`: Jupyter notebooks for analysis.
- `src/`: Source code.
    - `losses/`: Custom loss implementations.
    - `models/`: Model architectures.
    - `utils/`: Utility functions.
- `tests/`: Unit tests.

## Setup

```bash
pip install uv
uv sync
```

## Usage

```bash
python src/train.py
```

## Data Download

### Get Google API Credentials

You need a Google OAuth client:

1. Go to Google Cloud Console https://console.cloud.google.com/apis/credentials

2. Create OAuth Client ID → Desktop App

3. Download the file and save it as: `credentials.json`

4. Place credentials.json in the same folder as your script.

### OAuth Consent Requirements

If your Google Cloud OAuth app is not verified, Google restricts access to “test users only”.

To allow login:

1. Go to Google Cloud Console

2. Open: API & Services → OAuth consent screen → Test users

3. Add the Gmail account you will use to authenticate.

If you skip this, Google will show:

This app has not completed the Google verification process.
Only developer-approved testers can use it.

### Run script to download data

```bash
# Download DiffIQA
python sadhi/utils/gdown_zip.py --folder 1vZehlUPDyDfo6Mq1K8pAMe3pcjqdDRht --destination data/diffiq
```

### Authenticate Google Drive
When you run the script for the first time, it will open a browser window to authenticate your Google account. Follow the instructions in the browser to complete the authentication process.

If you encounter an `HttpError 403` indicating that the 'Google Drive API has not been used in project ... or it is disabled', you need to enable the API, just read the error log and follow the instructions:

1. Visit the link provided in the error message (e.g., `https://console.developers.google.com/apis/api/drive.googleapis.com/overview?project=...`).
2. Enable the Google Drive API for your project.
3. Rerun the data download script.

After authentication, a file named token.pickle is created and reused automatically.