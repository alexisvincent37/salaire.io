import time
import random
import re
import logging
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ApecScraper:
    def __init__(self):
        self.driver = self._init_driver()
        self.wait = WebDriverWait(self.driver, 10)
        self.data = []
        self.urls = set()

    def _init_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def restart_driver(self):
        try:
            self.driver.quit()
        except:
            pass
        self.driver = self._init_driver()
        self.wait = WebDriverWait(self.driver, 10)

    @staticmethod
    def clean_text(text):
        return text.strip().replace("\n", " ") if text else None

    def get_detail_by_label(self, label_keyword):
        try:
            xpath = f"//h4[contains(text(), '{label_keyword}')]/following-sibling::span"
            element = self.driver.find_element(By.XPATH, xpath)
            return self.clean_text(element.text)
        except NoSuchElementException:
            return None

    def extract_salary_bounds(self, salary_text):
        if not salary_text or "Non spécifié" in salary_text:
            return None, None
        numbers = re.findall(r'\d+(?:[.,]\d+)?', salary_text)
        numbers = [float(n.replace(',', '.')) for n in numbers]
        if len(numbers) >= 2:
            return numbers[0], numbers[1]
        elif len(numbers) == 1:
            return numbers[0], numbers[0]
        return None, None

    def analyze_tags(self, tags_list):
        entreprise, contrat, lieu = "Non spécifié", None, None
        keywords_contrat = ['cdi', 'cdd', 'intérim', 'stage', 'alternance', 'apprentissage', 'freelance']
        reste_tags = []

        for t in tags_list:
            text_lower = t.lower()
            if not contrat and any(k in text_lower for k in keywords_contrat):
                contrat = re.sub(r'^\d+\s+', '', t)
            elif not lieu and re.search(r' - \d+$', t):
                lieu = t
            else:
                reste_tags.append(t)

        if reste_tags:
            entreprise = reste_tags[0]
        return entreprise, contrat, lieu

    def scrape_offer_details(self, url):
        try:
            self.driver.get(url)
            time.sleep(random.uniform(1.0, 2.0))
            
            try:
                titre = self.clean_text(self.driver.find_element(By.TAG_NAME, "h1").text)
            except:
                return None

            if not titre or "Rechercher" in titre:
                return None

            tags_text = []
            try:
                ul_list = self.driver.find_element(By.CSS_SELECTOR, "ul.details-offer-list, ul.at-tags-light")
                items = ul_list.find_elements(By.TAG_NAME, "li")
                tags_text = [self.clean_text(item.text) for item in items]
            except NoSuchElementException:
                pass

            entreprise, contrat, lieu = self.analyze_tags(tags_text)
            salary_text = self.get_detail_by_label("Salaire")
            
            return {
                "titre": titre,
                "entreprise": entreprise,
                "contrat": contrat,
                "lieu": lieu,
                "salaire_brut_texte": salary_text,
                "experience": self.get_detail_by_label("Expérience"),
                "statut": self.get_detail_by_label("Statut"),
                "metier": self.get_detail_by_label("Métier"),
                "secteur": self.get_detail_by_label("Secteur"),
                "etudes": self.get_detail_by_label("études"),
                "teletravail": self.get_detail_by_label("Télétravail"),
                "url_source": url
            }
        except Exception as e:
            logging.error(f"Erreur lors du scraping de {url}: {e}")
            return None

    def collect_urls(self, keywords, pages_per_keyword):
        for kw in keywords:
            logging.info(f"Recherche secteur : {kw}")
            for i in range(pages_per_keyword):
                url = f"https://www.apec.fr/candidat/recherche-emploi.html/emploi?motsCles={kw}&page={i}"
                try:
                    self.driver.get(url)
                    if i == 0:
                        try:
                            cookie_btn = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
                            )
                            cookie_btn.click()
                        except:
                            pass
                    
                    time.sleep(random.uniform(1.5, 2.5))
                    elements = self.driver.find_elements(By.XPATH, "//a[contains(@href, '/detail-offre/')]")
                    for e in elements:
                        href = e.get_attribute("href")
                        if href:
                            self.urls.add(href)
                except Exception as e:
                    logging.warning(f"Erreur sur la page {i} pour {kw}: {e}")

    def save_data(self, filename, intermediate=False):
        if not self.data:
            return
        df = pd.DataFrame(self.data)
        df[['salaire_min', 'salaire_max']] = df['salaire_brut_texte'].apply(
            lambda x: pd.Series(self.extract_salary_bounds(x))
        )
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        suffix = "(Intermédiaire)" if intermediate else "(Final)"
        logging.info(f"Sauvegarde effectuée : {len(df)} lignes {suffix}")

    def run(self, keywords, pages_per_keyword):
        try:
            self.collect_urls(keywords, pages_per_keyword)
            logging.info(f"Total des liens collectés : {len(self.urls)}")

            for i, url in enumerate(list(self.urls)):
                detail = self.scrape_offer_details(url)
                if detail:
                    self.data.append(detail)
                
                if (i + 1) % 50 == 0:
                    self.save_data("dataset_apec_progress.csv", intermediate=True)
                
                if (i + 1) % 200 == 0:
                    self.restart_driver()

            self.save_data("dataset_apec_final.csv")
        finally:
            self.driver.quit()

if __name__ == "__main__":
    KEYWORDS = ["Informatique", "Commercial", "Finance", "Marketing", "Ingénieur"]
    PAGES = 10
    
    scraper = ApecScraper()
    scraper.run(KEYWORDS, PAGES)
