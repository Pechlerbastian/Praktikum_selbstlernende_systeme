import os

from image_net_downloader import ImageNetDownloader
from bing_downloader import Bing


class ImageLoader:

    '''
    This class coordinates the downloading process for creation of test data. In the initialization,
    the downloading directory is created and ImageNetDownloader and BingDownloader are used
    @author: Bastian Pechler
    '''
    def __init__(self, full_imagenet_list):
        self.full_imagenet_list = full_imagenet_list
        self.classes_contained_in_imagenet = {}
        self.classes_to_be_downloaded_from_bing = {}
        self.image_folder = os.path.join('data/image_net', 'imagenet_images')
        self.image_net_downloader = ImageNetDownloader(self.classes_contained_in_imagenet, self.image_folder)
        if not os.path.isdir(self.image_folder):
            os.mkdir(self.image_folder)

    '''
    This function coordinates the downloading processes. The classes to be downloaded are divided in two seperate sets,
    the ones contained in the ImageNet dataset and the ones not contained there. For the first subset, ImageNetDownloader
    is used to load as many images as there are contained. If the amount should be too small (e.g. 200 wanted, but only 
    100 links viable), additional downloads are done using BingDownloader. For the second subset, download is done
    by just utilizing the bing API)
    '''
    def download_images(self, amount_images_per_class, classes_to_scrape, chrome_version, firefox_version,
                        amount_already_downloaded=0):
        # classes to be scraped are divided in two dicts
        # first subset will be loaded from imagenet, second one will be downloaded from bing
        for key in classes_to_scrape.keys():
            if self.full_imagenet_list.keys().__contains__(key):
                self.classes_contained_in_imagenet[key] = classes_to_scrape[key]
            else:
                self.classes_to_be_downloaded_from_bing[key] = classes_to_scrape[key]

        downloaded_amount_of_images_per_class = {}
        for key in classes_to_scrape:
            downloaded_amount_of_images_per_class[key] = 0
        downloaded_amount_of_images_per_class.update(self.image_net_downloader.load_from_image_net(amount_images_per_class
                                                                                              - amount_already_downloaded))
        for key in downloaded_amount_of_images_per_class.keys():
            if downloaded_amount_of_images_per_class[key] < amount_images_per_class:
                self.classes_to_be_downloaded_from_bing[key] = classes_to_scrape[key]
                self.classes_contained_in_imagenet.__delitem__(key)
        # do the bing download
        for key in self.classes_to_be_downloaded_from_bing.keys():
            # check if there are already downloads from imagenet, if so no need for as many downloads
            if key in downloaded_amount_of_images_per_class.keys():
                self.download(self.classes_to_be_downloaded_from_bing[key][0], key, chrome_version, firefox_version,
                              amount_images_per_class - downloaded_amount_of_images_per_class[key],
                              amount_already_downloaded=amount_already_downloaded)
            else:
                self.download(self.classes_to_be_downloaded_from_bing[key][0], key, chrome_version, firefox_version,
                              amount_images_per_class,
                              amount_already_downloaded=amount_already_downloaded)

    '''
    This function creates an instance of the bing downloader and passes how many images have already been downloaded 
    from ImageNet, so that the training data can be completed.
    @author: Bastian Pechler
    '''
    def download(self, query, label, chrome_version, firefox_version, limit=100, amount_already_downloaded=0):
        path = os.path.join(self.image_folder, label)
        if not os.path.exists(path):
            os.mkdir(path)
        bing = Bing(query, limit, path, chrome_version, firefox_version, 60, amount_already_downloaded)
        bing.run()