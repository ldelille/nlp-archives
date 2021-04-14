
import scrapy
from scrapy.item import Item, Field


class NewsArticle(Item):
    title = Field()
    date_published = Field()
    text = Field()
    tags = Field()
    description = Field()
    url = Field()
