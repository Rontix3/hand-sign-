import cv2
import mediapipe as mp
import time
import pyttsx3
import threading

# Mediapipe kézdetektor inicializálása
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Szövegfelolvasó inicializálása
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Beszédsebesség beállítása

# Globális változó a beszéd szinkronizálásához
speaking = False

def speak(text):
    """Felolvassa a megadott szöveget egy külön szálban, hogy ne akassza meg a fő programot."""
    global speaking
    if speaking:
        return  # Ha már beszél, ne indítsunk új hangot
    speaking = True
    threading.Thread(target=run_speech, args=(text,)).start()

def run_speech(text):
    """Futtatja a beszédet külön szálban."""
    global speaking
    engine.say(text)
    engine.runAndWait()
    speaking = False  # Amikor vége a beszédnek, visszaállítjuk

# Funkció a kézjelek felismeréséhez
def detect_hand_sign(landmarks):
    thumb_is_open = landmarks[4].x > landmarks[2].x
    index_is_open = landmarks[8].y < landmarks[5].y
    middle_is_open = landmarks[12].y < landmarks[9].y
    ring_is_open = landmarks[16].y < landmarks[13].y
    pinky_is_open = landmarks[20].y < landmarks[17].y

    if thumb_is_open and index_is_open and middle_is_open and ring_is_open and pinky_is_open:
        return "Open Palm"
    elif not thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Fist"
    elif not thumb_is_open and index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "One Finger"
    elif not thumb_is_open and index_is_open and middle_is_open and not ring_is_open and not pinky_is_open:
        return "Two Finger"
    elif thumb_is_open and index_is_open and not middle_is_open and not ring_is_open and pinky_is_open:
        return "L Sign"
    elif not thumb_is_open and index_is_open and middle_is_open and ring_is_open and not pinky_is_open:
        return "Three Fingers"
    elif not thumb_is_open and index_is_open and middle_is_open and ring_is_open and pinky_is_open:
        return "Four Fingers"
    elif thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Thumbs Up"
    else:
        return "Unknown Gesture"

# Kamera inicializálása
cap = cv2.VideoCapture(0)

# Kijelzési késleltetés változók
display_time = 0.5  # másodperc, ameddig egy jel látható marad
last_display_time = time.time()
last_hand_sign = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Kép átkonvertálása RGB-re, mivel Mediapipe ezt várja
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Ha találunk kezet a képen
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Kézjel detektálása
            hand_sign = detect_hand_sign(hand_landmarks.landmark)
            
            # Ha új jel van, vagy eltelt a kijelzési idő, frissítjük a kijelzést és kimondjuk a gesztust
            if hand_sign != last_hand_sign or (time.time() - last_display_time > display_time):
                last_display_time = time.time()
                last_hand_sign = hand_sign
                speak(hand_sign)  # Mondja ki a gesztust külön szálban

            # A kézpontok rajzolása a képernyőre
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Csak egyszer jelenítsük meg a szöveget, a késleltetési logikával
        cv2.putText(frame, last_hand_sign, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Képernyő frissítése
    cv2.imshow("Hand Gesture Recognizer", frame)

    # Kilépés az 'q' gomb lenyomásával
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Erőforrások felszabadítása
cap.release()
cv2.destroyAllWindows()
hands.close()
