"""
University Web Scraper - استخراج محتوا از دانشگاه‌های برتر

ابزارهای scraping برای استخراج خودکار محتوا از منابع دانشگاهی
"""

import requests
from bs4 import BeautifulSoup
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path
import hashlib
from datetime import datetime
import json
from .university_storage import storage

# امنیت
from .university_security import security_monitor

logger = logging.getLogger(__name__)

class UniversityScraper:
    """
    Web Scraper برای استخراج محتوا از وب‌سایت‌های دانشگاهی
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config['scraping']['user_agent']
        })
        self.timeout = config['scraping']['timeout']
        self.rate_limit = config['scraping']['rate_limit']
        self.retry_attempts = config['scraping']['retry_attempts']
        
        # Cache directory
        self.cache_dir = Path(config['storage']['cache_dir'])
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def fetch_url(self, url: str, retry: int = 0) -> Optional[str]:
        """
        دریافت محتوای یک URL با مدیریت خطا و retry
        
        Args:
            url: آدرس URL
            retry: تعداد تلاش مجدد
        
        Returns:
            محتوای HTML یا None
        """
        # چک دامنه مجاز
        if not security_monitor.domain_check(url):
            return None

        try:
            time.sleep(self.rate_limit)  # Rate limiting
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            logger.info(f"✓ Fetched: {url}")
            return response.text
            
        except requests.RequestException as e:
            logger.warning(f"✗ Error fetching {url}: {e}")
            
            if retry < self.retry_attempts:
                logger.info(f"  Retrying ({retry + 1}/{self.retry_attempts})...")
                return self.fetch_url(url, retry + 1)
            
            return None
    
    def extract_links(self, html: str, base_url: str, filter_pattern: Optional[str] = None) -> List[str]:
        """
        استخراج لینک‌ها از HTML
        
        Args:
            html: محتوای HTML
            base_url: URL پایه برای لینک‌های نسبی
            filter_pattern: الگوی فیلتر برای لینک‌ها
        
        Returns:
            لیست لینک‌های استخراج‌شده
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # تبدیل به URL کامل
            if href.startswith('http'):
                full_url = href
            elif href.startswith('/'):
                full_url = base_url.rstrip('/') + href
            else:
                continue
            
            # فیلتر
            if filter_pattern is None or filter_pattern in full_url:
                links.append(full_url)
        
        return list(set(links))  # حذف تکراری
    
    def extract_text_content(self, html: str) -> Dict:
        """
        استخراج محتوای متنی از HTML
        
        Returns:
            دیکشنری شامل عنوان، متن، و متادیتا
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # حذف اسکریپت‌ها و استایل‌ها
        for script in soup(["script", "style"]):
            script.decompose()
        
        # عنوان
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No Title"
        
        # متن اصلی
        text = soup.get_text(separator='\n', strip=True)
        
        # پاراگراف‌ها
        paragraphs = [p.get_text().strip() for p in soup.find_all('p')]
        
        # هدینگ‌ها
        headings = []
        for i in range(1, 7):
            for h in soup.find_all(f'h{i}'):
                headings.append({
                    'level': i,
                    'text': h.get_text().strip()
                })
        
        return {
            'title': title_text,
            'text': text,
            'paragraphs': paragraphs,
            'headings': headings,
            'length': len(text)
        }
    
    def extract_pdf_links(self, html: str, base_url: str) -> List[str]:
        """استخراج لینک‌های PDF از صفحه"""
        soup = BeautifulSoup(html, 'html.parser')
        pdf_links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.endswith('.pdf') or '.pdf?' in href:
                if href.startswith('http'):
                    pdf_links.append(href)
                elif href.startswith('/'):
                    pdf_links.append(base_url.rstrip('/') + href)
        
        return list(set(pdf_links))
    
    def cache_content(self, url: str, content: Dict) -> None:
        """ذخیره محتوا در cache"""
        # ایجاد نام فایل از URL
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{url_hash}.json"
        
        cache_data = {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'content': content
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  Cached: {cache_file.name}")
    
    def get_cached_content(self, url: str) -> Optional[Dict]:
        """بازیابی محتوا از cache"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{url_hash}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
    
    def scrape_page(self, url: str, use_cache: bool = True) -> Optional[Dict]:
        """
        استخراج کامل محتوای یک صفحه
        
        Args:
            url: آدرس صفحه
            use_cache: استفاده از cache
        
        Returns:
            دیکشنری شامل تمام محتوای استخراج‌شده
        """
        # بررسی cache
        if use_cache:
            cached = self.get_cached_content(url)
            if cached:
                logger.info(f"  From cache: {url}")
                return cached['content']
        
        # دریافت HTML
        html = self.fetch_url(url)
        if not html:
            return None
        
        # استخراج محتوا
        raw_content = self.extract_text_content(html)

        # اعمال سیاست امنیتی روی متن
        enforcement = security_monitor.enforce(url, raw_content['text'])
        if not enforcement['allowed']:
            security_monitor.log("blocked_page", "Page blocked by policy", "error", url)
            return None

        # جایگزینی متن با نسخه سانیت شده (در صورت وجود findings)
        raw_content['text'] = enforcement['sanitized_content']
        raw_content['security'] = enforcement['scan']
        content = raw_content
        
        # استخراج لینک‌ها
        content['links'] = self.extract_links(html, url)
        content['pdf_links'] = self.extract_pdf_links(html, url)
        content['url'] = url
        
        # ذخیره در cache
        self.cache_content(url, content)
        
        return content


class UniversityResourceCollector:
    """
    جمع‌آوری منابع از دانشگاه‌های مختلف
    """
    
    def __init__(self, universities: Dict, config: Dict):
        self.universities = universities
        self.scraper = UniversityScraper(config)
        self.collected_data = {}
    
    def collect_from_university(
        self,
        university_key: str,
        max_pages: int = 10
    ) -> Dict:
        """
        جمع‌آوری محتوا از یک دانشگاه
        
        Args:
            university_key: کلید دانشگاه (مثلا "MIT")
            max_pages: حداکثر تعداد صفحات
        
        Returns:
            دیکشنری شامل تمام داده‌های جمع‌آوری‌شده
        """
        if university_key not in self.universities:
            logger.error(f"University not found: {university_key}")
            return {}
        
        university = self.universities[university_key]
        logger.info(f"\n{'='*80}")
        logger.info(f"Collecting from: {university['name']}")
        logger.info(f"{'='*80}")
        
        collected = {
            'university': university['name'],
            'resources': {},
            'total_pages': 0,
            'total_pdfs': 0
        }
        
        # جمع‌آوری از هر منبع
        for resource_key, resource_info in university['resources'].items():
            logger.info(f"\n  Resource: {resource_key}")
            logger.info(f"  URL: {resource_info['url']}")
            
            # استخراج صفحه اصلی
            content = self.scraper.scrape_page(resource_info['url'])
            
            if content:
                # Persist page in DB
                uni_id = storage.upsert_university(university_key, university)
                resource_id = storage.get_resource_id(uni_id, resource_key)
                if resource_id is None:
                    resource_id = storage.insert_resource(uni_id, resource_key, resource_info)
                storage.insert_page(resource_id, {
                    'url': content.get('url'),
                    'title': content.get('title'),
                    'length': content.get('length')
                })
                collected['resources'][resource_key] = {
                    'info': resource_info,
                    'main_page': content,
                    'sub_pages': []
                }
                
                collected['total_pages'] += 1
                collected['total_pdfs'] += len(content.get('pdf_links', []))
                
                # استخراج چند صفحه فرعی (محدود)
                sub_links = content.get('links', [])[:max_pages]
                
                for i, link in enumerate(sub_links, 1):
                    if i > max_pages:
                        break
                    
                    sub_content = self.scraper.scrape_page(link)
                    if sub_content:
                        storage.insert_page(resource_id, {
                            'url': sub_content.get('url'),
                            'title': sub_content.get('title'),
                            'length': sub_content.get('length')
                        })
                        collected['resources'][resource_key]['sub_pages'].append(sub_content)
                        collected['total_pages'] += 1
                        collected['total_pdfs'] += len(sub_content.get('pdf_links', []))
        
        logger.info(f"\n  Summary:")
        logger.info(f"  Total pages: {collected['total_pages']}")
        logger.info(f"  Total PDFs found: {collected['total_pdfs']}")
        
        self.collected_data[university_key] = collected
        return collected
    
    def collect_from_all(self, max_pages_per_resource: int = 5) -> Dict:
        """
        جمع‌آوری از همه دانشگاه‌ها
        
        Args:
            max_pages_per_resource: حداکثر صفحات هر منبع
        
        Returns:
            دیکشنری شامل داده‌های همه دانشگاه‌ها
        """
        for university_key in self.universities.keys():
            try:
                self.collect_from_university(university_key, max_pages_per_resource)
            except Exception as e:
                logger.error(f"Error collecting from {university_key}: {e}")
        
        return self.collected_data
    
    def get_statistics(self) -> Dict:
        """آمار جمع‌آوری"""
        total_pages = sum(data['total_pages'] for data in self.collected_data.values())
        total_pdfs = sum(data['total_pdfs'] for data in self.collected_data.values())
        
        return {
            'universities_collected': len(self.collected_data),
            'total_pages': total_pages,
            'total_pdfs': total_pdfs,
            'by_university': {
                key: {
                    'pages': data['total_pages'],
                    'pdfs': data['total_pdfs']
                }
                for key, data in self.collected_data.items()
            }
        }
