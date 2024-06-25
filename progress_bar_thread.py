import time
import threading
from traitlets import TraitError


class ProgressBarThread(threading.Thread):
    '''
    This function initialises the class from the Thread super class and saves the current Interaction
    Controller instance
    @author: Marcel Achner
    '''
    def __init__(self, frontend_model):
        super(ProgressBarThread, self).__init__()
        self.frontend_model = frontend_model

    '''
    This function runs a thread that is displaying the progress bar in the frontend when the model is retrained or 
    images are downloaded
    @author: Marcel Achner
    '''
    def run(self):
        while self.frontend_model.loading_active:
            try:
                self.frontend_model.loading_progress_value = 0
                for i in range(10):
                    time.sleep(0.25)
                    self.frontend_model.loading_progress_value += 1
            except ValueError:
                continue
            except TraitError:
                continue
    '''
    This function stops the running thread displaying the progress bar
    @author: Marcel Achner
    '''
    def stop(self):
        self.frontend_model.loading_active = False
