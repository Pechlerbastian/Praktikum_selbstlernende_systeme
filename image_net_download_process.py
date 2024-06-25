import glob
import multiprocessing
import os
import random
from multiprocessing import Value
from pathlib import Path

import requests
import time
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL

IMAGENET_API_WNID_TO_URLS = lambda \
        wnid: f'https://image-net.org/api/imagenet.synset.geturls?wnid={wnid}'


class ImageNetDownloadProcess(multiprocessing.Process):

    def __init__(self, classes_to_scrape, shared_amount_images_downloaded, amount_images_per_class, images_folder):
        self.imagenet_images_folder = images_folder
        self.downloaded_amount_per_class = shared_amount_images_downloaded
        self.classes_to_scrape = classes_to_scrape
        self.class_folder = ''
        self.class_images = Value('d', 0)
        self.images_per_class = amount_images_per_class
        super().__init__()

    '''
    This function is just delegating and is needed for Thread implementation.
    @author: Bastian Pechler
    '''
    def run(self):
        self.load_data_from_sublist()

    '''
    This function sends a request to get urls of the images from imagenet. Therefor the wnid is passed and a list of 
    available images is generated, which are then used to download the test data.
    @author: Bastian Pechler
    '''
    def load_data_from_sublist(self):
        for class_wnid in self.classes_to_scrape.keys():
            self.class_images.value = 0
            self.downloaded_amount_per_class[class_wnid] = 0
            url_urls = IMAGENET_API_WNID_TO_URLS(class_wnid)

            time.sleep(3)
            resp = requests.get(url_urls)

            self.class_folder = os.path.join(self.imagenet_images_folder, class_wnid)
            if not os.path.exists(self.class_folder):
                os.mkdir(self.class_folder)
            image_paths = glob.glob(self.class_folder + '*')
            self.class_images.value = len(image_paths)

            urls = [url.decode('utf-8') for url in resp.content.splitlines()]
            random.shuffle(urls)
            self.get_images(urls, class_wnid)

    '''
    This function downloads the training data by iterating over the url list of images of one class. If enough images 
    have been loaded it is stopped. It also checks if the images have the right data format and if the urls are
    working.
    @author: Bastian Pechler
    '''
    def get_images(self, img_url_list, class_wnid):
        try:
            for img_url in img_url_list:
                if len(img_url) <= 1:
                    continue

                if self.class_images.value > self.images_per_class:
                    return

                img_name = img_url.split('/')[-1]
                img_name = img_name.split("?")[0]
                if img_name.split('.')[-1] not in ["jpe", "jpeg", "jfif", "tiff", "gif", "bmp", "png", "webp", "jpg"]:
                    continue
                img_file_path = Path(os.path.join(self.class_folder, img_name))
                # this if is there to verify different images if download is started again
                if img_file_path.is_file():
                    continue
                try:
                    img_resp = requests.get(img_url, timeout=1)
                except ConnectionError:
                    continue
                except ReadTimeout:
                    continue
                except TooManyRedirects:
                    continue
                except MissingSchema:
                    continue
                except InvalidURL:
                    continue
                if 'content-type' not in img_resp.headers:
                    continue
                if 'image' not in img_resp.headers['content-type']:
                    continue
                if len(img_resp.content) < 1000:
                    continue
                with open(img_file_path, 'wb') as img_f:
                    img_f.write(img_resp.content)
                self.class_images.value += 1

        finally:
            self.downloaded_amount_per_class[class_wnid] = self.class_images.value
            return
