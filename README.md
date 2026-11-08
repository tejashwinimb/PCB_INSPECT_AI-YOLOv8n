🔬 PCB INSPECT AI

AI-Based PCB Defect Detection Using YOLOv8n

About the Project

PCB INSPECT AI is a project focused on making Printed Circuit Board (PCB) inspection easier and more efficient using Artificial Intelligence.

In traditional PCB manufacturing, identifying defects through manual inspection can take time and effort. To explore a smarter approach, I developed an AI-based system using the YOLOv8n object detection model.

The system allows users to upload a PCB image and automatically identify possible defects. It also displays the detected defects, their confidence scores, and an overall inspection summary through an interactive dashboard.

What Can It Detect?

The model is trained to identify six types of PCB defects:

- Missing Hole
- Mouse Bite
- Open Circuit
- Short
- Spur
- Spurious Copper

How It Works

1. Upload an image of a Printed Circuit Board.
2. The trained YOLOv8n model analyses the image.
3. The system identifies and marks the detected defects.
4. The dashboard displays the defect details and confidence scores.
5. An inspection summary provides an overview of the board.

Technologies Used

- Python
- YOLOv8n
- Ultralytics
- Streamlit
- OpenCV
- Pandas
- Plotly

Model Training and Results

The model was trained using a PCB defect dataset. The repository includes the training results, performance graphs, and confusion matrices used to understand and evaluate the model's performance.

Live Demo

You can try the application here:

👉 "PCB INSPECT AI – Open Live App" (https://tejashwinimb-pcb-inspect-ai-yolov8n-app-xwgyme.streamlit.app/)

Run the Project Locally

Clone this repository:

git clone https://github.com/tejashwinimb/PCB_INSPECT_AI-YOLOv8n.git
cd PCB_INSPECT_AI-YOLOv8n

Install the required libraries:

pip install -r requirements.txt

Run the application:

streamlit run app.py

Project Objective

The main objective of this project is to explore how deep learning can support automated PCB quality inspection and help make defect identification faster and more accessible.

---

Built as an engineering project exploring AI, computer vision, and automated PCB inspection.
