import multiprocessing
from image_net_download_process import ImageNetDownloadProcess
import os
from boltons import iterutils


class ImageNetDownloader:

    def __init__(self, classes_to_scrape, images_folder):
        self.amount_images_per_class = 0
        self.classes_to_scrape = classes_to_scrape
        self.images_folder = images_folder
        self.shared_amount_images_downloaded = multiprocessing.Manager().dict()
        if not os.path.isdir(self.images_folder):
            os.mkdir(self.images_folder)

    '''
    This function checks the amount of available cores of the current system, splits the label list to be downloaded in
    parts and creates different download threads to improve speed of the download.
    @author: Bastian Pechler
    '''
    def load_from_image_net(self, amount_images_per_class=20):
        self.amount_images_per_class = amount_images_per_class
        # if less classes than cores, we need less processes than classes
        amount_cores = min(multiprocessing.cpu_count() - 1, len(self.classes_to_scrape))
        processes = []

        length = int(len(self.classes_to_scrape) / amount_cores) + 1
        keys = [*self.classes_to_scrape.keys()]

        key_sub_lists = iterutils.chunked(keys, length)
        for i in range(amount_cores):
            if i >= len(key_sub_lists):
                break
            classes_sub_lists = {key: self.classes_to_scrape[key] for key in key_sub_lists[i]}
            processes.append(ImageNetDownloadProcess(classes_sub_lists, self.shared_amount_images_downloaded,
                                                     self.amount_images_per_class, self.images_folder))
        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()

        # this is needed to find those classes where too many urls have been defect
        # with this information it is possible to make additional downloads via bing
        downloaded_images_per_class = {}

        downloaded_images_per_class.update(self.shared_amount_images_downloaded)

        return downloaded_images_per_class
