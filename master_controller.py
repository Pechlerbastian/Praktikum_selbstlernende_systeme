from companion_model import CompanionModel
from interaction_controller import InteractionController, ProgressBarThread
from backend_model import BackendModel
from speech_to_text_controller import SpeechToTextController
from avatar_manager import AvatarManager
import threading


class MasterController:

    """
    This constructor call is instantiating all the needed backend model and all the controllers needed to use
    the functionality of this application.
    Needed controllers: AvatarManager, InteractionController, SpeechToTextController
    Needed models:  BackendModel
                        -> for state maintenance (amount classifiers, amount train data downloaded,
                                                  is model trained, have labels been modified, current label list)

                    CompanionModel
                        -> to load weights of current keras model, reset, (re)train, predict
    If weights will not fit the current label list (e.g. label was added, but interruption in training),
    the companion model will be reset to make it work again.
    @author: Bastian Pechler
    """
    def __init__(self, frontend_model, node_process):
        avatar_manager = AvatarManager()
        frontend_model = frontend_model

        frontend_model.loading_active = True
        frontend_model.loading_progress_description = "Downloading: "
        progress_bar_thread = ProgressBarThread(frontend_model=frontend_model)
        thread = threading.Thread(target=progress_bar_thread.start)
        thread.start()

        # This here is needed, if images are downloaded in Initialization of backend model!
        backend_model = BackendModel(frontend_model.number_download_images,
                                     frontend_model.model_chrome_version, frontend_model.model_firefox_version,
                                     node_process)
        progress_bar_thread.stop()
        thread.join()

        frontend_model.loading_active = True
        frontend_model.loading_progress_description = "Training: "
        progress_bar_thread = ProgressBarThread(frontend_model=frontend_model)
        thread = threading.Thread(target=progress_bar_thread.start)
        thread.start()
        try:
            companion_model = CompanionModel(amount_outputs=backend_model.output_classifiers,
                                             already_trained=backend_model.model_already_trained,
                                             learning_rate=frontend_model.model_learning_rate)
        except :
            print("Oops, it seems like the saved weights did not match your specified labels. "
                  "To fix this, your labels and model will be reset.")
            backend_model.reset()
            backend_model = BackendModel(frontend_model.number_download_images,
                                         frontend_model.model_chrome_version, frontend_model.model_firefox_version,
                                         node_process=node_process)
            companion_model = CompanionModel(amount_outputs=backend_model.output_classifiers,
                                             already_trained=backend_model.model_already_trained,
                                             learning_rate=frontend_model.model_learning_rate,
                                             model_path='model_backup.h5')

        progress_bar_thread.stop()
        thread.join()
        self.speech_to_text_controller = SpeechToTextController(frontend_model, backend_model, companion_model,
                                                                avatar_manager)
        self.button_interaction_controller = InteractionController(frontend_model, backend_model, companion_model,
                                                                   avatar_manager)
        return
