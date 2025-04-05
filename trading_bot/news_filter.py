import requests
from datetime import datetime, timedelta
from config_loader import Config

class NewsFilter:
    """Filter untuk memeriksa apakah ada berita fundamental berdampak tinggi."""

    def __init__(self):
        self.api_url = Config.NEWS_API_URL
        self.api_key = Config.NEWS_API_KEY

    def check_high_impact(self) -> bool:
        """
        Mengecek apakah ada berita berdampak tinggi dalam 3 jam terakhir.
        Menghindari trading saat berita penting muncul.
        """
        try:
            now = datetime.utcnow()
            params = {
                'q': 'gold OR XAU OR Fed OR inflation',
                'from': (now - timedelta(hours=24)).strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'apiKey': self.api_key,
                'language': 'en',
                'pageSize': 20
            }

            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            news_data = response.json()

            keywords = ['gold', 'XAU', 'interest rate', 'inflation', 'Fed', 'NFP']
            for article in news_data.get('articles', []):
                title = article.get('title', '').lower()
                if any(keyword in title for keyword in keywords):
                    pub_time = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                    if (now - pub_time).total_seconds() < 3 * 3600:
                        print(f"📢 Berita Penting Terbaru: {article['title']}")
                        return True

            return False
        except Exception as e:
            print(f"❌ Gagal memeriksa berita: {e}")
            return False
