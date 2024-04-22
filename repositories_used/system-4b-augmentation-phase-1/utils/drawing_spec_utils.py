import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
# help(mp_face_mesh.FaceMesh)

# Load drawing_utils and drawing_styles
import utils.mp_drawing_utils as mp_drawing
# #mp_drawing = mp.solutions.drawing_utils 

annotated_image_rescale = 4

mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=3, color=[255,255,0])
drawing_spec_landmarks = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=[255,0,0], size=0.1*annotated_image_rescale) #size=0.2
