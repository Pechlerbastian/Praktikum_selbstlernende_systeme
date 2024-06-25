import speech_recognition as sr
import time
from interaction_controller import InteractionController


class SpeechToTextController(InteractionController):

    def __init__(self, frontend_model, backend_model, companion_model, avatar_manager):
        super().__init__(frontend_model, backend_model, companion_model, avatar_manager)
        self.recognizer = sr.Recognizer()
        self.stop_idle = None

        self.microphone = sr.Microphone()
        if not self.backend_model.model_already_trained:
            self.handle_retrain()
        self.recognize_in_background()


    '''
    This callback is called repeatedly is a background thread, until the speech recognition is turned off.
    It uses different boolean flags to enable smooth voice interaction and asks for verification to ensure no wrong 
    functions are called due to wrong recognized words.
    '''
    def activation_callback(self, recognizer, audio):  # this is called from the background thread
        try:
            recorded_text = recognizer.recognize_google(audio_data=audio, language='en-US')
        except sr.UnknownValueError:
            return
        except sr.WaitTimeoutError:
            self.frontend_model.sound_recognizer_active = 'hidden'
            self.frontend_model.sound_recognizer_not_active = 'visible'
            self.stop_idle(wait_for_stop=True)
            return

        if recorded_text == "":
            return

        elif 'hello' in recorded_text:
            self.handle_hello()
            return

        elif 'bye' in recorded_text:
            self.handle_goodbye()
            return

        elif 'detect' in recorded_text or 'detection' in recorded_text or 'detecting' in recorded_text or 'recognize' in recorded_text:
            self.handle_webcam_started()
            return

        elif 'what' in recorded_text and 'is' in recorded_text:
            self.detect()
            return

        elif 'why' in recorded_text and 'think' in recorded_text:
            self.handle_asked_for_explanation()
            return

        elif 'show' in recorded_text and 'synonym' in recorded_text:
            self.show_synonym()
            return

        elif 'no' in recorded_text and self.frontend_model.do_not_let_user_validate_result == 'visible' and \
                not self.backend_model.label_to_be_specified and not self.backend_model.can_specify_alternative_label:
            self.frontend_model.do_not_let_user_enter_label = 'visible'

            if 'that' in recorded_text or 'those' in recorded_text or 'this' in recorded_text or 'it' in recorded_text:
                self.frontend_model.input_label = recorded_text.split(' ')[-1]
                self.backend_model.probable_label_to_be_verified = True
                self.handle_label_input()
                return

            else:
                self.handle_label_to_be_specified()
                return

        elif self.backend_model.probable_label_to_be_verified and (
                'yes' in recorded_text or ('correct' in recorded_text and 'not' not in recorded_text)
                or 'true' in recorded_text):
            self.frontend_model.do_not_let_user_validate_label = 'hidden'
            self.frontend_model.do_not_let_user_validate_result = 'hidden'
            self.backend_model.label_to_be_specified = False
            self.backend_model.probable_label_to_be_verified = False
            self.handle_user_labelling()
            return

        elif self.backend_model.probable_label_to_be_verified and ('no' in recorded_text or 'false' in recorded_text):
            self.backend_model.probable_label_to_be_verified = False
            self.backend_model.label_to_be_specified = True
            return

        elif self.backend_model.label_to_be_specified and len(recorded_text) >= 2 and \
                ('this' in recorded_text or 'those' in recorded_text):
            self.frontend_model.input_label = recorded_text.split(' ')[-1]
            self.handle_label_input()
            self.frontend_model.do_not_let_user_validate_label = 'visible'
            self.frontend_model.do_not_let_user_validate_result = 'visible'
            self.backend_model.label_to_be_specified = False
            self.backend_model.probable_label_to_be_verified = True
            return

        elif 'download' in recorded_text and 'retrain' in recorded_text:
            self.handle_perceived_download_and_retrain_request()
            return

        elif 'retrain' in recorded_text:
            self.handle_perceived_retrain_request()
            return

        elif self.backend_model.download_and_retrain_to_be_verified and \
                ('correct' in recorded_text or 'yes' in recorded_text):
            self.handle_download_and_retrain()
            return

        elif self.backend_model.retrain_to_be_verified and 'correct' in recorded_text:
            self.handle_retrain()
            return

        elif 'yes' in recorded_text and self.frontend_model.do_not_let_user_validate_result == 'visible':
            self.backend_model.can_specify_alternative_label = True
            self.frontend_model.output_text = 'You can specify another label, if you want.'
            self.handle_speak_request()
            return

        elif ('label' in recorded_text or 'synonym') and self.backend_model.can_specify_alternative_label:
            self.backend_model.add_synonym(recorded_text.split(' ')[-1])
            self.frontend_model.output_text = 'Thank you for the feedback. The added label is ' \
                                              + recorded_text.split(' ')[-1]
            self.backend_model.can_specify_alternative_label = False
            self.frontend_model.do_not_let_user_validate_result = 'hidden'
            self.frontend_model.do_not_let_user_validate_label = 'hidden'
            self.backend_model.label_list_modified = True
            self.backend_model.save_state()
            self.handle_speak_request()
            return

        elif 'no' in recorded_text and self.backend_model.can_specify_alternative_label:
            self.handle_correct_prediction()
            self.backend_model.can_specify_alternative_label = False
            return

        elif 'reset' in recorded_text and 'model' in recorded_text:
            self.reset()
            return

    '''
    This function closes the background thread listening to voice commands. It also calls the super implementation and
    gives some feedback to the user.
    @author: Bastian Pechler
    '''
    def handle_goodbye(self):
        super().handle_goodbye()
        try:
            self.stop_idle(wait_for_stop=True)
        except Exception:
            return
            # this exception will be thrown because thread kills itself
            # no other workaround needed, because listening is stopped here

    '''
    This function is only needed for communication by voice. Fritzi will ask the user 
    if the recognized label is correct.
    @author: Bastian Pechler
    '''
    def handle_label_input(self):
        self.frontend_model.output_text = 'The shown object was labeled as ' + self.frontend_model.input_label + \
                                          '. \n Please verify the displayed label!'
        self.handle_speak_request()

    '''
    This function handles the user request to record an image from the webcam feed. After the picture is recorded and
    displayed in the frontend, the user gets a response, describing how to interact further.
    @author: Bastian Pechler
    '''
    def handle_webcam_started(self):
        super().handle_webcam_started()
        self.frontend_model.output_text = 'Start the object detection with the phrase \"What is this?\"'
        self.handle_speak_request()

    '''
    This function handles the user request asking for an explanation of the prediction. 
    @author: Bastian Pechler
    '''
    def handle_asked_for_explanation(self):
        self.explain_prediction()
        self.frontend_model.output_text = 'Here are the recognized patterns'
        self.handle_speak_request()

    '''
    This function lets the application respond to a greeting formula (like hello or hi)
    @author: Bastian Pechler
    '''
    def handle_hello(self):
        self.frontend_model.output_text = 'Hello'
        self.backend_model.avatar_active = True
        if self.backend_model.avatar_active:
            self.speak_avatar_dialect('Hi, I am Fritzi, what can I do for you sir?')

    '''
    This function improves readability of the code. It checks if the sound recognizer is active, 
    by using the visibility parameter saved in the GUI model. 
    author: Bastian Pechler
    '''
    def sound_recognizer_active(self):
        return self.frontend_model.sound_recognizer_active == 'visible'

    '''
    This function adding a boolean flag, so that a user can be asked for the correct label.
    @author: Bastian Pechler
    '''
    def handle_label_to_be_specified(self):
        self.backend_model.label_to_be_specified = True
        super().handle_label_to_be_specified()

    '''
    This function is called when the sound recognizer perceives a download and retrain request. In turn the user is 
    asked for verification.
    author: Bastian Pechler
    '''
    def handle_perceived_download_and_retrain_request(self):
        self.backend_model.download_and_retrain_to_be_verified = True
        self.frontend_model.output_text = 'Verify download and retrain.'
        self.handle_speak_request()
        self.frontend_model.output_text = self.frontend_model.output_text + ' You can do so by saying "correct" ' \
                                                                            'or "yes"'
    '''
    This function is called when the sound recognizer perceives a retrain request. In turn the user is 
    asked for verification.
    author: Bastian Pechler
    '''
    def handle_perceived_retrain_request(self):
        self.backend_model.retrain_to_be_verified = True
        self.frontend_model.output_text = 'Please verify, if you want to retrain.'
        self.handle_speak_request()
        self.frontend_model.output_text = self.frontend_model.output_text + ' You can do so by saying "correct" ' \
                                                                            'or "yes"'

    '''
    This function is called when the sound recognizer perceives a download and retrain request. In turn the user is 
    asked for verification.
    author: Bastian Pechler
    '''
    def handle_download_and_retrain(self):
        super().handle_download_and_retrain()
        self.backend_model.download_and_retrain_to_be_verified = False

    '''
    This function is called when the detection is finished. It calls the super implementation (of InteractionController)
    and adds a additional output text, so that the user knows how to continue with the converstaion.
    author: Bastian Pechler
    '''
    def handle_detection_finished(self):
        super().handle_detection_finished()
        self.frontend_model.output_text = self.frontend_model.output_text + \
                                          ' You can validate by saying "correct" or "wrong", ' \
                                          'or by clicking the according button'

    '''
    This function handles a retrain request by the user. It calls the super implementation (of InteractionController)
    and manages the state in SpeechRecognizer.
    author: Bastian Pechler
    '''
    def handle_retrain(self):
        super().handle_retrain()
        self.backend_model.retrain_to_be_verified = False

    '''
     This function starts the background thread for speech recognition and adjusts for ambient noise.
     The callback call is called repeatedly until self.stop_idle is called.
     author: Bastian Pechler
     '''
    def recognize_in_background(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        self.stop_idle = self.recognizer.listen_in_background(self.microphone, self.activation_callback,
                                                              phrase_time_limit=10)
