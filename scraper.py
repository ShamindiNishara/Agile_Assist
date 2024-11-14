import os
import requests
import urllib.parse

def download_pdfs(links_file, download_folder, failed_log="failed_downloads.txt"):
    # Ensure the download folder exists
    os.makedirs(download_folder, exist_ok=True)

    # Read all links from the file
    with open(links_file, 'r') as file:
        links = [line.strip() for line in file if line.strip()]

    # Clear or create the failed downloads log file
    open(failed_log, 'w').close()

    # Download each PDF
    for i, link in enumerate(links):
        print(f"PDF {i} of {len(links)}")
        try:
            
            # Make the HTTP GET request
            response = requests.get(link, stream=True)
            response.raise_for_status()  # Check if the request was successful

            # Write the PDF content to a file
            try:
                # Decode the URL and use the last part as the filename
                pdf_name = urllib.parse.unquote(link.split("/")[-1])
                file_path = os.path.join(download_folder, pdf_name)
                with open(file_path, 'wb') as pdf_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        pdf_file.write(chunk)
            except Exception:
                # Decode the URL and use the last part as the filename
                pdf_name = f"PDF_{i}.pdf"
                file_path = os.path.join(download_folder, pdf_name)
                with open(file_path, 'wb') as pdf_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        pdf_file.write(chunk)

            print(f"Downloaded: {pdf_name}")

        except requests.exceptions.RequestException as e:
            print(f"\nFailed to download {link}: {e}\n")
            # Append failed URL to the log file
            with open(failed_log, 'a') as fail_file:
                fail_file.write(f"{link}\n")

# Usage
links_file = "links.txt"  # Path to your text file with links
download_folder = "pdf_downloads"  # Folder to save PDFs
failed_log = "failed_downloads.txt"  # File to log failed downloads
download_pdfs(links_file, download_folder, failed_log)
