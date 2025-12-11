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
    If the file already exists locally, skip downloading.
    """
    request = service.files().get_media(fileId=file_id)
    local_path = Path(output_dir) / file_name
    # Ensure parent directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if file already exists
    if local_path.is_file():
        print(f"Skipping {file_name}, already exists at {local_path}")
        return

    fh = io.FileIO(local_path, mode="wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"Downloading {file_name}: {int(status.progress() * 100)}%", end="\r")
    print(f"\nDownloaded: {file_name} → {local_path}")


def download_zip_files_recursive(folder_id, output_dir="downloads"):
    """
    Recursively download all .zip files from the specified Google Drive folder and its subfolders.
    The directory structure in Google Drive is mirrored locally under `output_dir`.
    """
    service = get_service()
    # Ensure the base output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _download_from_folder(current_folder_id, current_path):
        # List items in the current folder
        items = list_files_in_folder(service, current_folder_id)
        if not items:
            return
        # Separate zip files and subfolders
        zip_items = [i for i in items if i["name"].lower().endswith(".zip")]
        subfolders = [
            i for i in items if i["mimeType"] == "application/vnd.google-apps.folder"
        ]

        # Download zip files into the current_path
        for z in zip_items:
            file_id = z["id"]
            file_name = z["name"]
            download_file(service, file_id, file_name, current_path)

        # Recurse into subfolders
        for sub in subfolders:
            sub_id = sub["id"]
            sub_name = sub["name"]
            sub_path = Path(current_path) / sub_name
            sub_path.mkdir(parents=True, exist_ok=True)
            _download_from_folder(sub_id, str(sub_path))

    # Start recursion from the root folder
    _download_from_folder(folder_id, output_dir)


# Backward compatible alias
download_zip_files_from_folder = download_zip_files_recursive


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
