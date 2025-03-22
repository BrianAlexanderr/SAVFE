# SAVFE - Smart Face Verification

SAVFE is a simple web-based application for userface verification using face recognition. This system captures user faces and if the face is verified then redirect them.

## 🚀 Features
- Face recognition-based student attendance
- AI-powered diagnosis feature for medical assessments
- Role-based login system (Doctor & Patient)
- Django backend with Firebase authentication
- Chat system between doctors and patients
- History tracking for diagnoses

## 🛠 Installation

### 1️⃣ Clone the Repository

`git clone https://github.com/BrianAlexanderr/SAVFE.git`<br>
`cd SAVFE`

### 2️⃣ Install Dependencies
Ensure you have Python installed, then install dependencies from requirements.txt:

`pip install -r requirements.txt`

### 3️⃣ Run the Server

`python manage.py runserver`<br>
Open http://127.0.0.1:8000/ in your browser.

### 📂 Project Structure
SAVFE/
│── BackEnd&Model/    # AI models & backend logic<br>
│── SAVFE HTML/       # Frontend templates<br>
│── manage.py         # Django project entry point<br>
│── requirements.txt  # Required dependencies<br>
└── README.md         # Project documentation<br>
<br><br>
### 🤖 AI Model Usage
The AI model is stored separately and must be downloaded before running the project. The model files (.keras) exceed GitHub's file size limit and are stored using Git LFS.
<br><br>
#### Download the Model:
Request the model files from the project owner.
<br><br>
Place them in BackEnd&Model/.
<br><br>
### 📜 License
This project is licensed under the MIT License.


