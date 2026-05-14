import json
import os
import time

import requests


def download_pdf(url, save_dir, filename):
    try:
        os.makedirs(save_dir, exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"File saved to: {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Download failed due to network error: {str(e)}")
    except IOError as e:
        print(f"File save failed: {str(e)}")
    except Exception as e:
        print(f"An unknown error occurred: {str(e)}")
    return None


def get_file_size(file_path):
    return os.path.getsize(file_path)


def download_pdf_from_set(file_path, out_path):
    paper_ids = set()
    for data in open(file_path, "r", encoding="utf-8"):
        d = json.loads(data)
        paper_ids.add(d["id"])
    print(f"Total papers: {len(paper_ids)}")
    download_times, sizes = [], []
    for item in paper_ids:
        pdf_url = f"https://openreview.net/pdf?id={item}"
        new_filename = f"{item}.pdf"
        start_time = time.time()
        file_path = download_pdf(pdf_url, out_path, new_filename)
        end_time = time.time()
        duration = end_time - start_time
        if file_path and os.path.isfile(file_path):
            size = get_file_size(file_path)
            sizes.append(size)
            download_times.append(duration)
            print(
                f"Downloaded {item} in {duration:.2f} seconds, size: {size/1024/1024:.2f} MB"
            )
        time.sleep(3)
    if sizes:
        avg_time = sum(download_times) / len(download_times)
        avg_size = sum(sizes) / len(sizes)
        print(
            f"Total downloads: {len(sizes)}, average download time: {avg_time:.2f} seconds, average paper size: {avg_size/1024/1024:.2f} MB"
        )


if __name__ == "__main__":
    download_pdf_from_set("./benchmark/test.json", "./benchmark/pdf/test")
