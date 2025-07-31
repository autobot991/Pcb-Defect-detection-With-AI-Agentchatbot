import os
import tempfile
import requests
import cv2 as cv
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from ultralytics import YOLO

# Load .env variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_DIR = './pcb.pt'

defect_names_map = {
    0: "Missing hole",
    1: "Mouse bite",
    2: "Open circuit",
    3: "Short",
    4: "Spur",
    5: "Supurious copper"
}

# --- Groq Chat Function ---
def ask_groq(messages, model="llama-3.3-70b-versatile", temperature=0.7):
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"\u26a0\ufe0f Groq Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Request failed: {e}"

# --- Main App ---
def main():
    global model
    model = YOLO(MODEL_DIR)

    # Sidebar: Defect Class Info
    st.sidebar.header("🔍 PCB Defect Classes")

    for defect in defect_names_map.values():
        st.sidebar.markdown(f"- *{defect}*")

    # Sidebar: AI Chat Agent
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Chat with PCB Expert")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a senior electronics manufacturing engineer and AI assistant specializing in Printed Circuit Board (PCB) defects. You can explain types of defects, suggest root causes, offer testing techniques, and answer both technical and beginner-level questions."}
        ]

    predefined_questions = [
        "What causes mouse bite defects in PCBs?",
        "How can open circuits be tested?",
        "What is the impact of short defects?",
        "How do you fix missing holes in a PCB?",
        "Explain spurious copper and how it occurs.",
        "What inspection techniques are used for PCBs?",
        "How can PCB defects affect signal integrity?",
        "What is the difference between open and short circuits?",
        "How does automated optical inspection (AOI) work?",
        "What materials are most prone to spur defects?",
        "Why is missing hole a critical defect?",
        "Can PCB defects be repaired or must they be discarded?"
    ]

    selected_q = st.sidebar.selectbox("📘 Example Questions", [""] + predefined_questions)

    user_input = st.sidebar.text_input("Type your question...")

    if st.sidebar.button("Ask Example") and selected_q:
        user_input = selected_q

    if st.sidebar.button("Ask AI") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.sidebar.spinner("Thinking..."):
            reply = ask_groq(st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.sidebar.markdown(f"**AI:** {reply}")

    for msg in st.session_state.chat_history[1:]:
        role = "🧑‍🔧 You" if msg["role"] == "user" else "🤖 AI"
        st.sidebar.markdown(f"**{role}:** {msg['content']}")

    # Main Title and File Uploader
    st.title("🔧 PCB Defect Detection with Ai Agents")
    st.write("Upload a PCB image or video to detect defects and generate summaries.")

    uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4'])

    if uploaded_file:

        # --- AI Chat Agent: Full Chat Interface ---
        st.markdown("---")
        st.header("🤖 PCB AI Chat")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "system", "content": "You are a helpful AI assistant who answers questions about PCB defects like 'mouse bite', 'short circuit', 'open circuit', and related topics in manufacturing and inspection."}
            ]

        user_message = st.chat_input("Ask the AI anything about PCB defects...")

        if user_message:
            st.session_state.chat_history.append({"role": "user", "content": user_message})
            with st.spinner("AI is thinking..."):
                reply = ask_groq(st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

        for msg in st.session_state.chat_history[1:]:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["content"])

        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = [
                {"role": "system", "content": "You are a helpful AI assistant who answers questions about PCB defects like 'mouse bite', 'short circuit', 'open circuit', and related topics in manufacturing and inspection."}
            ]
        if uploaded_file.type.startswith('image'):
            inference_images(uploaded_file)
        elif uploaded_file.type.startswith('video'):
            inference_video(uploaded_file)

# --- Image Inference ---
def inference_images(uploaded_file):
    image = Image.open(uploaded_file)
    results = model.predict(image)
    boxes = results[0].boxes
    plotted = results[0].plot()[:, :, ::-1]

    if len(boxes) == 0:
        st.markdown("**✅ No Defects Detected**")
    else:
        defect_indices = boxes.cls.cpu().numpy()
        defect_names = [defect_names_map[int(i)] for i in defect_indices if int(i) in defect_names_map]
        defect_summary = {d: defect_names.count(d) for d in set(defect_names)}
        defect_summary_str = ', '.join([f"{k}: {v}" for k, v in defect_summary.items()])

        st.markdown(f"**🔍 Total Defects Detected:** {len(defect_names)}")
        st.markdown("**📋 Defect Summary:** " + defect_summary_str)
        st.image(plotted, caption="Detected Defects", width=600)

        if st.button("🧠 Summarize This Image"):
            messages = [
                {"role": "system", "content": "You are a PCB inspection expert."},
                {"role": "user", "content": f"Summarize the following PCB defect detections in this image: {defect_summary_str}"}
            ]
            summary = ask_groq(messages)
            st.success("📄 Summary:")
            st.markdown(summary)

        for defect in set(defect_names):
            if st.button(f"❓ Learn about {defect}"):
                messages = [
                    {"role": "system", "content": "You are a PCB defect expert."},
                    {"role": "user", "content": f"Tell me more about the '{defect}' defect in PCBs. Include causes, detection techniques, and how to fix it."}
                ]
                reply = ask_groq(messages)
                st.markdown(f"**ℹ️ About {defect}:**\n\n{reply}")

# --- Video Inference ---
def inference_video(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    cap = cv.VideoCapture(temp_file.name)
    frame_count = 0
    defect_names_total = []

    if not cap.isOpened():
        st.error("❌ Error opening video file.")
        return

    frame_placeholder = st.empty()
    stop_placeholder = st.button("Stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 == 0:
            results = model.predict(frame, conf=0.75)
            plotted = results[0].plot()
            defect_indices = results[0].boxes.cls.cpu().numpy()
            defect_names = [defect_names_map[int(i)] for i in defect_indices if int(i) in defect_names_map]
            defect_names_total.extend(defect_names)

            if defect_names:
                summary_dict = {d: defect_names.count(d) for d in set(defect_names)}
                summary_str = ', '.join(f"{k}: {v}" for k, v in summary_dict.items())
                frame_placeholder.image(plotted, channels="BGR", caption=f"Detected Defects: {summary_str}")

            if stop_placeholder:
                break

    cap.release()
    os.unlink(temp_file.name)

    if defect_names_total:
        total_summary = {d: defect_names_total.count(d) for d in set(defect_names_total)}
        defect_summary_str = ', '.join([f"{k}: {v}" for k, v in total_summary.items()])
        st.markdown("**📋 Full Video Defect Summary:** " + defect_summary_str)

        if st.button("🧠 Summarize This Video"):
            messages = [
                {"role": "system", "content": "You are a PCB inspection expert."},
                {"role": "user", "content": f"Summarize the following defect detections from the video: {defect_summary_str}"}
            ]
            summary = ask_groq(messages)
            st.success("📄 Video Summary:")
            st.markdown(summary)

# --- Launch ---
if __name__ == '__main__':
    main()