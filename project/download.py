import bs4
import requests
from typing import AnyStr

class Download:
    """
    Use:
    download = Download()
    download.download("https://physionet.org/physiobank/database/bidmc/bidmc_csv/")
    to download all the data into the <data> folder
    """
    @staticmethod
    def download(url: AnyStr):
        page = requests.get(url)
        data = bs4.BeautifulSoup(page.text, "html.parser")
        for link in data.find_all("a"):
            file = link["href"]

            if file.endswith(".csv") or file.endswith(".txt"):
                with open("./data/" + file, "wb") as f:
                    print("writing: " + file)
                    r = requests.get(url + file)
                    f.write(r.content)
