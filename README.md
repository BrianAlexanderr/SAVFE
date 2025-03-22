# SAVFE - Smart Attendance Verification using Facial Evaluation

SAVFE is a web-based application for student attendance verification using face recognition. This system captures student faces and matches them against a pre-stored database to ensure accurate attendance tracking.

## 🚀 Features
- Face recognition-based student attendance
- AI-powered diagnosis feature for medical assessments
- Role-based login system (Doctor & Patient)
- Django backend with Firebase authentication
- Chat system between doctors and patients
- History tracking for diagnoses

## 🛠 Installation

### 1️⃣ Clone the Repository

`git clone https://github.com/BrianAlexanderr/SAVFE.git`
`cd SAVFE`

### 2️⃣ Install Dependencies
Ensure you have Python installed, then install dependencies from requirements.txt:

`pip install -r requirements.txt`

### 3️⃣ Run the Server

`python manage.py runserver`
Open http://127.0.0.1:8000/ in your browser.

### 📂 Project Structure
SAVFE/
│── BackEnd&Model/    # AI models & backend logic
│── SAVFE HTML/       # Frontend templates
│── manage.py         # Django project entry point
│── requirements.txt  # Required dependencies
└── README.md         # Project documentation

### 🤖 AI Model Usage
The AI model is stored separately and must be downloaded before running the project. The model files (.keras) exceed GitHub's file size limit and are stored using Git LFS.

#### Download the Model:
Request the model files from the project owner.

Place them in BackEnd&Model/.

📜 License
This project is licensed under the MIT License.


