import cv2
from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm
import mediapipe as mp

def detect_emotions(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='retinaface')
        if result:
            dominant_emotion = result[0]['dominant_emotion']
            return dominant_emotion, result[0]['region']
    except Exception as e:
        print(f"Erro no DeepFace: {e}")
    return None, None

import math

def calculate_angle(a, b, c):
    """
    Calcula o ângulo entre três pontos: a, b, c.
    a, b, c são tuples no formato (x, y, z).
    """
    ab = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def categorize_activities(pose_landmarks, mp_pose):
    if not pose_landmarks:
        return "Indefinido"

    # Extraindo coordenadas importantes
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Calcular ângulos
    angle_left_elbow = calculate_angle(
        (left_shoulder.x, left_shoulder.y, left_shoulder.z),
        (left_elbow.x, left_elbow.y, left_elbow.z),
        (left_wrist.x, left_wrist.y, left_wrist.z),
    )
    angle_right_elbow = calculate_angle(
        (right_shoulder.x, right_shoulder.y, right_shoulder.z),
        (right_elbow.x, right_elbow.y, right_elbow.z),
        (right_wrist.x, right_wrist.y, right_wrist.z),
    )

    # Identificar apertos de mão
    dist_between_wrists = math.sqrt(
        (left_wrist.x - right_wrist.x) ** 2 +
        (left_wrist.y - right_wrist.y) ** 2 +
        (left_wrist.z - right_wrist.z) ** 2
    )

    if dist_between_wrists < 0.1 and abs(angle_left_elbow - angle_right_elbow) < 20:
        return "Aperto de mão"

    # Identificar dança (movimento amplo de braços e pernas)
    if angle_left_elbow > 120 or angle_right_elbow > 120:
        return "Dançando"
    
    # Cálculos básicos de ângulos ou posições
    left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    left_eye = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE.value]
    left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    left_arm_up = left_elbow.y < left_eye.y
    right_arm_up = right_elbow.y < right_eye.y

    # Exemplos básicos de categorizações
    if left_knee.y > left_hip.y and right_knee.y > right_hip.y:
        return "Sentado"
    elif left_arm_up or right_arm_up:
        return "Acenando"
    else:
        return "Indefinido"

def detect_face_and_activities(video_path, output_video_path, output_text_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    emotion_color_map = {
        'happy': (0, 255, 0),
        'sad': (255, 0, 0),
        'angry': (0, 0, 255),
        'surprise': (255, 255, 0),
        'neutral': (255, 255, 255)
    }

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    anomaly_count = 0  # Variável para contar anomalias detectadas

    for frame_idx in tqdm(range(total_frames), desc="Processando emoções e atividades humanas em vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar emoções
        emotion, face_region = detect_emotions(frame)
        if face_region:
            x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
            color = emotion_color_map.get(emotion, (36, 255, 12))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

        # Converter o frame para RGB para o MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Detectar atividades com base na pose
        activity = "Indefinido"
        if results.pose_landmarks:
            activity = categorize_activities(results.pose_landmarks, mp_pose)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Verificar anomalias (considerar "Indefinido" como uma anomalia)
        if activity == "Indefinido":
            anomaly_count += 1

        # Adicionar texto de atividade detectada
        cv2.putText(frame, f"Atividade: {activity}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Salvar informações do frame no arquivo de texto
        with open(output_text_path, 'a', encoding='utf-8') as file:
            text = f"Frame {frame_idx + 1}: \nEmoção reconhecida: {emotion}\nAtividade Reconhecida: {activity}\n\n"
            file.write(text)

        # Escrever frame processado
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Escrever resumo final no arquivo de texto
    with open(output_text_path, 'a', encoding='utf-8') as file:
        file.write(f"\nResumo do vídeo:\nTotal de frames: {total_frames}\nAnomalias detectadas: {anomaly_count}\n")

def analyse_video(video_path):
    detect_face_and_activities(video_path, './output_video.mp4', './output_text.txt')

if __name__ == "__main__":
    analyse_video('../Unlocking Facial Recognition_ Diverse Activities Analysis.mp4')
