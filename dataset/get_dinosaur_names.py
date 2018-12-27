import scrapy

class SpiderSauraus(scrapy.Spider):
    name = 'spidersauraus'
    start_urls = ['https://en.wikipedia.org/wiki/List_of_dinosaur_genera']

    def parse(self, response):
        filename = 'dinosaurs.txt'
        dinos = set()
        count = 0
        with open(filename, 'w') as f:
            for dino in response.css('ul>li'):
                dino_name = dino.css('i > a ::text').extract_first()
                if dino_name != None:
                    dinos.add(dino_name)
                    if (count+1) == len(dinos):
                        f.write(dino_name)
                        f.write('\n')
                        count += 1
        print ('{} Dinosaurs found!'.format(count))
