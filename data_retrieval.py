import requests
import os
import zipfile
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

Source1 = "http://www.jsbach.net/midi/index.html"
oriSource1 = "http://www.jsbach.net/midi/"

def get_soup(source_url):
    """
    retrieve and parse from the website
    """
    try:
            
        r1 = requests.get(source_url)
        r1.raise_for_status()  # Raise an HTTPError for bad responses
        soup = BeautifulSoup(r1.content, 'html.parser')
        return soup.find_all('a', href=True)

    except:
        print("error")
        return None


midi_metadata = []

midi_folder = "downloads"

os.makedirs(midi_folder, exist_ok=True)

def extract_bwv_name(filename, parent_url,isZip):
    bwv_match = re.search(r"bwv[_\-]?(\d+)([_\-]?.*)?", filename, re.IGNORECASE)
    # print("When the name has:",bwv_match)
    if not bwv_match or isZip:
        bwv_match = re.search(r"bwv[_\-]?(\d+)([_\-]?.*)?", parent_url, re.IGNORECASE)
        # print("When the parent has:",bwv_match)
    
    if bwv_match:

        number = bwv_match.group(1).zfill(4) 
        extra = bwv_match.group(2) or ""  
        extra = extra.strip("_-") 
        name = f"bwv{number}_{extra}".strip("_").lower()
    else:

        name = os.path.splitext(filename)[0].lower()

    return name

def download_file(url, parent_url, is_zip=False):

    try:
        response = requests.get(url, stream=True)
        filename = os.path.basename(urlparse(url).path)

        if is_zip:
            zip_path = os.path.join(midi_folder, filename)
            with open(zip_path, 'wb') as f:
                f.write(response.content)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for extracted_file in zip_ref.namelist():
                    if extracted_file.endswith('.mid'):
                        extracted_filename = os.path.basename(extracted_file)
                        new_filename = extract_bwv_name(extracted_filename, parent_url,is_zip)
                        new_filepath = os.path.join(midi_folder, new_filename)
                        with open(new_filepath, 'wb') as midi_file:
                            midi_file.write(zip_ref.read(extracted_file))
                        # Add metadata
                        download_metadata.append({
                            "name": new_filename,
                            "url": url,
                            "parent_url": parent_url
                        })

            os.remove(zip_path)  
        else:
            
            new_filename = extract_bwv_name(filename, parent_url,is_zip)
            filepath = os.path.join(midi_folder, new_filename)

            with open(filepath, 'wb') as f:
                f.write(response.content)


            download_metadata.append({
                "name": new_filename,
                "url": url,
                "parent_url": parent_url
            })

        print("downloaded mids")
    except:
        print("error ")


files1 = get_soup(Source1)

#album links
albums = set()
for i in files1:
    subSource1 = i['href']
    albums.add(subSource1)


for j in albums:
    
    if "http://" in j or "/../" in j:
        continue

    full_url = urljoin(oriSource1, j)
    
    files2 = get_soup(full_url)

    for k in files2:
        href_file = k['href']
        if href_file.endswith('.mid'):
            final_url = urljoin(full_url, href_file)
            download_file(final_url, full_url)

        elif href_file.endswith('.zip'):
            final_url = urljoin(full_url, href_file)
            download_file(final_url, full_url, is_zip=True)

# saving metadata
with open("download_metadata.json", "w") as json_file:
    json.dump(download_metadata, json_file, indent=4)

print("done")
