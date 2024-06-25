import time
import threading
from webcam_processor import WebcamProcessor
from data_parser import DataParser
from progress_bar_thread import ProgressBarThread


class InteractionController:

    '''
    This method instantiates the interaction controller (used for button clicks).
    Needed objects:
        CompanionModel
        WebcamProcessor
        FrontendModel
        BackendModel
        AvatarManager
    @author: Bastian Pechler
    '''
    def __init__(self, frontend_model, backend_model, companion_model, avatar_manager):
        self.companion_model = companion_model
        self.frontend_model = frontend_model
        self.backend_model = backend_model
        self.webcam_processor = WebcamProcessor()
        self.avatar_manager = avatar_manager

    '''
    This function handles the deactivation of the voice recognizer
    @author: Bastian Pechler
    '''
    def handle_goodbye(self):
        self.frontend_model.sound_recognizer_active = 'hidden'
        self.frontend_model.sound_recognizer_not_active = 'visible'
        if self.backend_model.avatar_active:
            self.speak_avatar_dialect('Talk to you soon!')

    '''
    This function delegates a request to start the webcam input feed and shows a adequate response to the user
    @author: Bastian Pechler
    '''
    def handle_webcam_started(self):
        self.frontend_model.output_text = 'Select a section in the image by pointing at the object!'
        self.handle_speak_request()
        self.bounding_box()

    '''
    This function shows the user that detection was done. The detected label is displayed.
    @author: Bastian Pechler
    '''
    def handle_detection_finished(self):
        self.frontend_model.input_label = self.frontend_model.output_text
        self.frontend_model.output_text = "The object is a " + self.frontend_model.output_text + \
                                          '. Please validate this label now.'
        self.frontend_model.do_not_let_user_validate_result = 'visible'
        self.handle_speak_request()

    '''
    This method calls the explanation feature of our model.
    author: Bastian Pechler
    '''
    def handle_asked_for_explanation(self):
        self.explain_prediction()
        self.frontend_model.output_text = 'Here are the recognized patterns'
        self.handle_speak_request()

    '''
    This method checks if the avatar is activated. If so, it will read the content of 
    the output textfield (located in the GUI). 
    @author Bastian Pechler
    '''
    def handle_speak_request(self):
        if self.backend_model.avatar_active:
            self.speak_avatar_dialect(self.frontend_model.output_text)

    '''
    This method is called if the user specified the prediction as wrong. In this case, the correct label has to 
    be specified, so that the model can retrain or add another label to the label list. 
    @author: Bastian Pechler
    '''
    def handle_label_to_be_specified(self):
        self.frontend_model.output_text = 'Please specify which label could be used for the shown object!'
        self.handle_speak_request()

    '''
    This method is called if the user wished the classification of a detection of an (uploaded or recorded) image.
    @author: Bastian Pechler
    '''
    def detect(self):
        if len(self.frontend_model.image_bytes) == 0:
            self.frontend_model.output_text = 'Please provide me an input picture first.'
            self.handle_speak_request()
        else:
            self.backend_model.current_label_index = self.companion_model.predict(self.frontend_model.image_path)
            key = [*self.backend_model.label_list.keys()][self.backend_model.current_label_index]
            self.backend_model.list_entry = 0
            self.frontend_model.output_text = self.backend_model.label_list[key][self.backend_model.list_entry]
            self.frontend_model.do_not_let_user_validate_result = 'visible'
            self.handle_detection_finished()

    '''
    This method is there that all classification labels for one class of objects can be viewed. 
    After all those synonyms are shown once, it will display that all labels have been shown.
    @author: Bastian Pechler
    '''
    def show_synonym(self):
        if self.frontend_model.do_not_let_user_validate_result == 'hidden':
            self.frontend_model.output_text = 'Please let me classify the image first.'

        elif self.backend_model.current_label_index is not None:
            key = [*self.backend_model.label_list.keys()][self.backend_model.current_label_index]
            if len(self.backend_model.label_list[key]) - 1 > self.backend_model.list_entry:
                self.backend_model.list_entry += 1
                self.frontend_model.output_text = 'A known synonym is :' + self.backend_model.label_list[key][
                    self.backend_model.list_entry]
            else:
                self.backend_model.list_entry = 0
                self.frontend_model.output_text = 'No more synonyms for the current label are known.'

        self.handle_speak_request()

    '''
    This function checks if entered label is new to the companion model (not yet in label list)
    In this case, additional training data is generated by downloading images.
    If the label is already present, but the model predicted wrong it is just retrained
    @author: Bastian Pechler
    '''
    def handle_user_labelling(self):
        entered_label = self.frontend_model.input_label.lower()
        new_label_key = self.backend_model.check_user_label(entered_label)
        if new_label_key is not None:
            self.frontend_model.output_text = 'Downloading and retraining to make this object ' \
                                              'detectable in the future'
            if self.backend_model.avatar_active:
                self.handle_speak_request()
            self.frontend_model.loading_active = True
            self.frontend_model.loading_progress_description = "Downloading: "
            progress_bar_thread = ProgressBarThread(frontend_model=self.frontend_model)
            thread = threading.Thread(target=progress_bar_thread.start)
            thread.start()
            self.backend_model.generate_data_for_new_label(self.frontend_model.image_bytes, entered_label,
                                                           new_label_key, self.frontend_model.model_chrome_version,
                                                           self.frontend_model.model_firefox_version)
            self.companion_model.add_output_for_classifier(len(self.backend_model.label_list))
            self.backend_model.output_classifiers = len(self.backend_model.label_list)
            DataParser.save_current_state(self.backend_model.amount_downloaded_images_per_class,
                                          self.backend_model.model_already_trained,
                                          self.backend_model.label_list_modified,
                                          self.backend_model.output_classifiers)
            progress_bar_thread.stop()
            thread.join()
        self.frontend_model.output_text = 'The training process will start now!'
        if self.backend_model.avatar_active:
            self.handle_speak_request()
        self.retrain_model()
        self.frontend_model.output_text = 'Training is done!'
        if self.backend_model.avatar_active:
            self.handle_speak_request()

    '''
    This function handles a retrain request by the user
    @author: Bastian Pechler
    '''
    def handle_retrain(self):
        self.frontend_model.output_text = 'Retraining, this may take a while.'
        self.handle_speak_request()
        time.sleep(3)
        self.retrain_model()

    '''
    This function handles a retrain and download request by the user
    @author: Bastian Pechler
    '''
    def handle_download_and_retrain(self):
        self.frontend_model.output_text = 'Downloading and retraining, this can take a while'
        self.handle_speak_request()
        self.download_and_retrain()

    '''
    This function is called when a user classifies an image as correct. He/ She is thanked for the feedback
    @author: Bastian Pechler
    '''
    def handle_correct_prediction(self):
        self.frontend_model.output_text = 'Thank you for the feedback'
        self.handle_speak_request()

    '''
    This function delegates a retrain request to the companion model.
    Data is parsed (by DataParser), UI Elements are managed and the new state is persisted (model trained = true)
    @author: Bastian Pechler
    '''
    def retrain_model(self):
        self.frontend_model.loading_active = True
        self.frontend_model.loading_progress_description = "Training: "
        progress_bar_thread = ProgressBarThread(frontend_model=self.frontend_model)
        thread = threading.Thread(target=progress_bar_thread.start)
        thread.start()

        self.frontend_model.do_not_let_user_validate_result = 'hidden'
        self.frontend_model.do_not_let_user_enter_label = 'hidden'
        generator, list_key_to_int = self.backend_model.parse_images_and_labels(
            self.frontend_model.image_augmentation_active, self.frontend_model.model_chunk_size)

        self.companion_model.set_generator(generator, list_key_to_int)

        self.companion_model.improvement_threshold = self.frontend_model.model_accuracy_improvement_threshold
        self.companion_model.train_model(self.frontend_model.model_learning_rate,
                                         self.frontend_model.model_accuracy_improvement_threshold,
                                         self.frontend_model.number_training_epochs)
        self.backend_model.model_already_trained = True
        DataParser.save_current_state(self.backend_model.amount_downloaded_images_per_class,
                                      self.backend_model.model_already_trained, self.backend_model.label_list_modified,
                                      self.backend_model.output_classifiers)
        self.frontend_model.output_text = "Done training"

        progress_bar_thread.stop()
        thread.join()


    '''
    This function resets the model to the initial state
    @author: Bastian Pechler
    '''
    def reset(self):
        self.frontend_model.output_text = 'The prediction model will be reset to the basic state.'
        self.handle_speak_request()
        self.companion_model.reset(self.frontend_model.model_learning_rate)
        self.backend_model.reset()

    '''
    This function delegates a download and retrain request. The amount of images downloaded will be set to the amount
    specified by the user in the frontend. If He/ she specifies more images than before, download is triggered. Else 
    just the state saved in backend is updated.
    Afterwards a retrain is triggered.
    @author: Bastian Pechler
    '''
    def download_and_retrain(self):
        self.frontend_model.loading_active = True
        self.frontend_model.loading_progress_description = "Downloading: "
        progress_bar_thread = ProgressBarThread(frontend_model=self.frontend_model)
        thread = threading.Thread(target=progress_bar_thread.start)
        thread.start()

        self.backend_model.download_images(self.frontend_model.model_chrome_version,
                                           self.frontend_model.model_firefox_version,
                                           amount=self.frontend_model.number_download_images
                                           )
        self.backend_model.download_images = self.frontend_model.number_download_images
        self.backend_model.save_state()

        progress_bar_thread.stop()
        thread.join()
        self.retrain_model()

    '''
    This function connects the frontend with the webcam processor to detect the objects on the webcam feed and 
    implement the fingerpointing functionality. The specified object is passed to the frontend and displayed there.
    @author: Marcel Achner
    '''
    def bounding_box(self):
        self.frontend_model.image_path = self.webcam_processor.detect_objects()
        # to use that you need to return the path of the screenshot that needs to be extracted to the net
        # if the path cannot be read the webcam was manually closed and nothing should happen with the displayed img
        try:
            file = open(self.frontend_model.image_path, "rb")
            self.frontend_model.image_bytes = file.read()
            self.backend_model.image_to_be_verified = True
        except OSError:
            pass

    '''
    This function calls the explain_prediction method in the backend to show why the generated label was chosen 
    by Fritzi
    @author: Marcel Achner
    '''
    def explain_prediction(self):
        self.frontend_model.loading_active = True
        self.frontend_model.loading_progress_description = "Explaining: "
        progress_bar_thread = ProgressBarThread(frontend_model=self.frontend_model)
        thread = threading.Thread(target=progress_bar_thread.start)
        thread.start()

        resulting_image_paths = self.companion_model.explain_prediction(self.frontend_model.image_path)
        self.frontend_model.img_xai_one_bytes = open(resulting_image_paths[0][0], "rb").read()
        self.frontend_model.img_xai_two_bytes = open(resulting_image_paths[1][0], "rb").read()

        progress_bar_thread.stop()
        thread.join()

    '''
    This function passes the desired dialect of the avatar from the frontend to the backend and speaks the specified 
    message
    @author: Marcel Achner
    '''
    def speak_avatar_dialect(self, message):
        self.avatar_manager.set_dialect(self.frontend_model.avatar_dialect)
        self.avatar_manager.avatar_speak(self.backend_model.node_process, message)



