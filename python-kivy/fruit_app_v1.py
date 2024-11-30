from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
from ultralytics import YOLO
from langchain_core.messages import HumanMessage
from PIL import Image as PIL_Image
import re
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import base64,io
from dotenv import load_dotenv

class PhotoApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        
        # Image display widget
        self.image_widget = Image()
        self.add_widget(self.image_widget)
        
        # Horizontal layout for buttons 
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        self.add_widget(button_layout)
        
        # Capture button to start video stream
        self.capture_button = Button(text="Capture")
        self.capture_button.bind(on_press=self.start_camera)
        button_layout.add_widget(self.capture_button)
        
        # Take Photo button to capture a frame
        self.take_photo_button = Button(text="Process")
        self.take_photo_button.bind(on_press=self.capture_photo)
        button_layout.add_widget(self.take_photo_button)
        
        
        # Camera capture setup
        self.camera = cv2.VideoCapture("http://192.168.110.146:8080/video")
        # self.camera = cv2.VideoCapture(0)
        self.captured_image = None
        self.camera_active = False

        load_dotenv()
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
        os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")


        self.model = YOLO('yolov8s.pt')
        genaimodel = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash-latest",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                )
        self.gen_ai_model=genaimodel

    def start_camera(self, instance):
        # Start displaying the video feed
        if not self.camera_active:
            self.camera_active = True
            Clock.schedule_interval(self.update_camera, 1/30)
    
    def update_camera(self, *args):
        # Capture and display camera frames in real-time
        if self.camera_active:
            ret, frame = self.camera.read()
            if ret:
                buf = cv2.flip(frame, 0).tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.image_widget.texture = texture

    def process_with_generative_model(self, object_img):
            pil_image = PIL_Image.fromarray(cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")  # Change "JPEG" to "PNG" if needed
            image_bytes = buffer.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            genai_prompt= "Analyze the given image and check if it contains any edible item. If edible, determine its freshness on a scale of 0 to 100, where 100 indicates excellent freshness and 0 indicates spoiled. Provide the result in the format: '<item>:<score>' (e.g., 'apple:90'). If the image does not contain any edible item, respond with 'non-eatable'. Assume the input images are primarily single objects detected by a YOLO model, focusing on Indian foods, especially South Indian items."
           

            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": genai_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            )
            response=self.gen_ai_model.invoke([message])


            print(response.content)
            return response.content

    def getColours(self,cls_num):
            base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            color_index = cls_num % len(base_colors)
            increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
            color = [base_colors[color_index][i] + increments[color_index][i] * 
            (cls_num // len(base_colors)) % 256 for i in range(3)]
            return tuple(color)
    
    def postprocess(self, detections, frame):
        orig_img=frame.copy()
        for result in detections:
            classes_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.4:
                    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                    cls = int(box.cls[0])
                    class_name = classes_names[cls]
                    colour = self.getColours(cls)

                    # Extract object image for the generative model
                    object_img = frame[y1:y2, x1:x2]
                    generative_output = self.process_with_generative_model(object_img)

                    # cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                    if generative_output:
                        cv2.putText(frame, generative_output, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , colour, 2)

                    try:
                        goodness=int(re.findall(r'\d+', generative_output)[0])                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), self.score_to_bgr(goodness), -1)
                        frame = cv2.addWeighted(orig_img, 0.5, frame, 1 - 0.5, 0) 
                    except Exception as e:
                        print(e)

        opacity = 0.5 
        # cv2.addWeighted(orig_img, opacity, frame, 1 - opacity, 0, frame)
        print("done")
        return frame


    def capture_photo(self, instance):
        # Capture and display a single frame
        if self.camera_active:
            ret, frame = self.camera.read()
            if ret:
                self.captured_image = frame.copy()
                self.camera_active = False  # Stop the live feed
                Clock.unschedule(self.update_camera)                
                self.display_image(self.blur_and_add_text(self.captured_image.copy()))
                self.process_photo(self.captured_image)

    def score_to_bgr(self,score):
        # Clamp the score to ensure it is within the range of 0 to 100
        score = max(0, min(score, 100))
        
        # Calculate red and green values
        red = int(255 * (1 - score / 100))  # Red decreases from 255 to 0
        green = int(255 * (score / 100))    # Green increases from 0 to 255
        blue = 0                             # No blue component

        # Return the color in BGR format
        return (blue, green, red)  # OpenCV uses BGR format


    def display_image(self, img):
        # Display a given image
        buf = cv2.flip(img, 0).tobytes()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture

    def blur_and_add_text(self,image):
        # App`ly a Gaussian blur to the image
        blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
        
        # Define the text and font settings
        text = "Processing"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_color = (255, 255, 255)  # White text
        thickness = 3
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate the position for the text to be centered
        text_x = (blurred_image.shape[1] - text_size[0]) // 2
        text_y = (blurred_image.shape[0] + text_size[1]) // 2
        
        # Put the text on the blurred image
        cv2.putText(blurred_image, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        return blurred_image

    def process_photo(self, instance):
        # Flip and display the captured image
        tmp_img=self.captured_image
        if self.captured_image is not None:   
            detections = self.model.track(self.captured_image, stream=True)
            output_frame = self.postprocess(detections, self.captured_image)

            # Display the frame in the Kivy interface
            buf = cv2.flip(output_frame, 0).tostring()
            texture = Texture.create(size=(output_frame.shape[1], output_frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = texture
            self.display_image(output_frame)

    def on_stop(self):
        # Release the camera resource on app stop
        self.camera.release()

class PhotoAppApp(App):
    def build(self):
        return PhotoApp()

if __name__ == '__main__':
    PhotoAppApp().run()
