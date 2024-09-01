import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup


def scrape_text_from_(url):
    """
    Scrap the text contained on the web page located at the URL link.

    :parameter
    ----------
    url : str
        L'URL de la page web à scraper.

    :return
    -------
    str
        Le contenu textuel extrait de la page.
    """
    try:
        # make request to the website:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the text from web page:
        text = soup.get_text(separator='\n', strip=True)
        return text

    except RequestException as e:
        print(f"Erreur lors de la requête : {e}")


def main():
    """
    Main function
    """
    # Exemple d'utilisation
    # url = "https://www.example.com"
    url = "https://github.com/marta1994/efficient_bpe_explanation"
    scraped_text = scrape_text_from_(url)
    if scraped_text:
        # print("Contenu textuel extrait :")
        print(scraped_text)


if __name__ == '__main__':
    main()
