from __future__ import print_function

import argparse
import io
import os
import pickle
from pathlib import Path

import google.auth.exceptions
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# If modifying scopes, delete the file token.pickle.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_service():
    """
    Authenticate and return a Drive API service instance.
    Uses console-based OAuth flow (no GUI browser needed).
    """
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    # If no valid credentials, start the OAuth flow.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except google.auth.exceptions.RefreshError:
                creds = None

        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            # Use console flow instead of local_server (no browser needed)
            # Older versions may not have run_console, so use run_local_server with open_browser=False
            try:
                creds = flow.run_local_server(open_browser=False)
            except Exception as e:
                print(f"Authentication failed: {e}")
                raise

        # Save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    from googleapiclient.discovery import build

    service = build("drive", "v3", credentials=creds)
    return service


def list_files_in_folder(service, folder_id):
    """
    List all files in a specific Google Drive folder.
    Returns a list of dicts: {'id': ..., 'name': ..., 'mimeType': ...}
    """
    files = []
    page_token = None

    # Query: files whose parent is the given folder and are not trashed
    q = f"'{folder_id}' in parents and trashed = false"

    while True:
        response = (
            service.files()
            .list(
                q=q,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
            )
            .execute()
        )

        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if page_token is None:
            break

    return files


def download_file(service, file_id, file_name, output_dir):
    """
    Download a file from Google Drive by file_id to output_dir/file_name.
    """
    request = service.files().get_media(fileId=file_id)
    local_path = Path(output_dir) / file_name
    local_path.parent.mkdir(parents=True, exist_ok=True)

    fh = io.FileIO(local_path, mode="wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"Downloading {file_name}: {int(status.progress() * 100)}%", end="\r")
    print(f"\nDownloaded: {file_name} → {local_path}")


def download_zip_files_from_folder(folder_id, output_dir="downloads"):
    """
    Download only .zip files from the given Google Drive folder.
    Skips all other file types.
    """
    service = get_service()
    files = list_files_in_folder(service, folder_id)

    if not files:
        print("No files found in this folder.")
        return

    print(f"Found {len(files)} file(s) in folder. Checking for .zip files...")

    zip_files = [f for f in files if f["name"].lower().endswith(".zip")]

    if not zip_files:
        print("No .zip files found in this folder.")
        return

    print(f"Found {len(zip_files)} .zip file(s). Starting download...\n")

    for f in zip_files:
        file_id = f["id"]
        file_name = f["name"]
        download_file(service, file_id, file_name, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download only .zip files from a Google Drive folder."
    )
    parser.add_argument("folder_id", help="Google Drive folder ID to scan")
    parser.add_argument(
        "destination",
        nargs="?",
        default=".",
        help="Local directory to save zip files (default: current directory)",
    )
    args = parser.parse_args()

    os.makedirs(args.destination, exist_ok=True)

    download_zip_files_from_folder(args.folder_id, args.destination)
