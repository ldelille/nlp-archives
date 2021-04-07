import scrapy
from scrapy.loader import ItemLoader

from ScrapingNews.items import NewsArticle


# scrapy crawl lemonde - O ArticleLeMondeRun3 / europe.json - a keyword = asie - a depth = 300 - a year_min = 1990 - a year_max = 2000


class LeMondeSpiderSingle(scrapy.Spider):
    name = "lemonde_single"

    def start_requests(self):
        url = getattr(self, 'url', None)
        yield scrapy.Request(url, self.parse)

    def parse(self, response):
        article = NewsArticle()
        temp_paragraph = []
        article['title'] = response.css('h1.article__title::text').get()
        article['description'] = response.css('p.article__desc::text').get()
        article['date_published'] = response.xpath(
            '//*[contains(concat( " ", @class, " " ), concat( " ", "meta__date--header", " " ))]/text()').get()
        article['url'] = response.request.url
        for paragraph in response.css('p.article__paragraph '):
            temp_paragraph.append(paragraph.xpath('string()').get())
        if len(temp_paragraph) > 2:
            article['text'] = ''.join(temp_paragraph)
            yield article
        else:
            return
