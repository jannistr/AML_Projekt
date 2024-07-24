"pip install flask opencv-python-headless pyautogui numpy mediapipe tensorflow"


#importieren der relevanten Bibliotheken
print("importing (1 of 8) modules...")
from flask import Flask, render_template, redirect, url_for
print("importing (2 of 8) modules...")
import threading
print("importing (3 of 8) modules...")
import cv2
print("importing (4 of 8) modules...")
import pyautogui
print("importing (5 of 8) modules...")
import numpy as np
print("importing (6 of 8) modules...")
import mediapipe as mp
print("importing (7 of 8) modules...")
import time
print("importing (8 of 8) modules...")
import tensorflow as tf
print("Starting WebApp...")
app = Flask(__name__)
process = None
algorithm_thread = None
stop_event = threading.Event()

# Algorithmus 1 als Funktion definieren
def algorithm1(stop_event):
    cap = cv2.VideoCapture(0)
    previous_frame = None
    #Schwellenwert zur Ausführung eines Sprunges
    jump_threshold = 50000
    last_jump_time = 0 
    jump_cooldown = 0.5

    def detect_jump(frame, previous_frame):
        #Binären Differenzframe bilden
        diff = cv2.absdiff(previous_frame, frame)
        _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        diff = cv2.dilate(diff, None, iterations=2)
        non_zero_count = np.count_nonzero(diff)
        return non_zero_count > jump_threshold

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Sprung ausführen, wenn sich beide Frames stark unterscheiden
        if previous_frame is not None:
            current_time = time.time()
            if detect_jump(gray, previous_frame) and (current_time - last_jump_time > jump_cooldown):
                print("Jump detected!")
                pyautogui.press('space')
                last_jump_time = current_time

        previous_frame = gray
        cv2.imshow('Frame', frame)
        # Zweite Methode zur Beendigung des Algorithmus durch Drücken von 'q' im Webcam-Fenster
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Algorithmus 1 als Funktion definieren
def algorithm2(stop_event):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    # Nutzt MediaPipe zur Erkennung jedes einzelnen Finger(gelenkes)
    def is_hand_open(hand_landmarks):
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
        index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
        # Wenn jeder der Finger über den Handknochen liegt, dann gilt dies als offene Hand
        open_hand = (thumb_tip.y < thumb_mcp.y and
                     index_finger_tip.y < index_finger_mcp.y and
                     middle_finger_tip.y < middle_finger_mcp.y and
                     ring_finger_tip.y < ring_finger_mcp.y and
                     pinky_tip.y < pinky_mcp.y)

        return open_hand

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        # Wenn eine offene Hand erkannt wird, wird der Sprung durchgeführt.
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if is_hand_open(hand_landmarks):
                    print("Sprung")
                    pyautogui.press('space')
                else:
                    print("Kein Sprung")

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Frame', frame)

        # Zweite Methode zur Beendigung des Algorithmus durch Drücken von 'q' im Webcam-Fenster
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Algorithmus 1 als Funktion definieren
def algorithm3(stop_event):
    # Laden des selbst-trainierten Modells
    gesture_model = tf.keras.models.load_model('gesture_model_own_data.h5')
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)

    def preprocess_frame(frame):
        resized = cv2.resize(frame, (64, 64))
        normalized = resized / 255.0
        normalized = np.expand_dims(normalized, axis=0)
        return normalized

    previous_prediction = None

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        # Detektierte Hand ausschneiden und dem Modell zur Klassifizierung übergeben
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                hand_frame = frame[y_min:y_max, x_min:x_max]

                if hand_frame.size == 0:
                    continue
                # Wenn das Modell eine offene Hand erkannt hat, dann Sprung ausführen.
                processed_frame = preprocess_frame(hand_frame)
                prediction = gesture_model.predict(processed_frame)[0][0]
                if previous_prediction is not None:
                    if previous_prediction <= 0.5 and prediction > 0.5:
                        print("Open hand detected! Jump!")
                        pyautogui.press('space')
                    elif previous_prediction > 0.5 and prediction <= 0.5:
                        print("Closed hand detected! No jump!")

                previous_prediction = prediction
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        # Zweite Methode zur Beendigung des Algorithmus durch Drücken von 'q' im Webcam-Fenster
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Routing zur Startseite
@app.route('/')
def index():
    return render_template('index.html')

# Routing zur game.html Seite mit dem Aufruf des ersten Algorithmus
@app.route('/game1')
def game1():
    global algorithm_thread, stop_event
    stop_event.set()
    if algorithm_thread is not None:
        algorithm_thread.join()
    stop_event.clear()
    algorithm_thread = threading.Thread(target=algorithm1, args=(stop_event,))
    algorithm_thread.start()
    return render_template('game.html')

# Routing zur game.html Seite mit dem Aufruf des zweiten Algorithmus
@app.route('/game2')
def game2():
    global algorithm_thread, stop_event
    stop_event.set()
    if algorithm_thread is not None:
        algorithm_thread.join()
    stop_event.clear()
    algorithm_thread = threading.Thread(target=algorithm2, args=(stop_event,))
    algorithm_thread.start()
    return render_template('game.html')

# Routing zur game.html Seite mit dem Aufruf des dritten Algorithmus
@app.route('/game3')
def game3():
    global algorithm_thread, stop_event
    stop_event.set()
    if algorithm_thread is not None:
        algorithm_thread.join()
    stop_event.clear()
    algorithm_thread = threading.Thread(target=algorithm3, args=(stop_event,))
    algorithm_thread.start()
    return render_template('game.html')

# laufenden Algorithmus abbrechen und zurückkehren zur Startseite
@app.route('/stop')
def stop():
    global algorithm_thread, stop_event
    stop_event.set()
    if algorithm_thread is not None:
        algorithm_thread.join()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
