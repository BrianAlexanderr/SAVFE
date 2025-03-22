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
```bash
git clone https://github.com/BrianAlexanderr/SAVFE.git
cd SAVFE
2️⃣ Install Dependencies
Ensure you have Python installed, then install dependencies from requirements.txt:

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Set Up the Database
Run database migrations:

bash
Copy
Edit
python manage.py migrate
4️⃣ Run the Server
bash
Copy
Edit
python manage.py runserver
Open http://127.0.0.1:8000/ in your browser.

📂 Project Structure
bash
Copy
Edit
SAVFE/
│── BackEnd&Model/    # AI models & backend logic
│── SAVFE HTML/       # Frontend templates
│── manage.py         # Django project entry point
│── requirements.txt  # Required dependencies
└── README.md         # Project documentation
🤖 AI Model Usage
The AI model is stored separately and must be downloaded before running the project. The model files (.keras) exceed GitHub's file size limit and are stored using Git LFS.

Download the Model:
Request the model files from the project owner.

Place them in BackEnd&Model/.

⚠️ Notes
Ensure Firebase Authentication is configured if using Firebase for login.

For AI diagnosis, the model needs to be trained on the appropriate dataset.

📜 License
This project is licensed under the MIT License.


