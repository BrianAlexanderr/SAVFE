function startScanning() {
    document.getElementById('welcome-screen').style.display = 'none';
    document.getElementById('scan-screen').style.display = 'block';
  
    // Simulate scanning delay
    setTimeout(() => {
      const granted = Math.random() > 0.5; // Randomly grant or deny access
      if (granted) {
        document.getElementById('scan-screen').style.display = 'none';
        document.getElementById('granted-screen').style.display = 'block';
      } else {
        document.getElementById('scan-screen').style.display = 'none';
        document.getElementById('denied-screen').style.display = 'block';
      }
    }, 3000); // 3 seconds delay
  }
  
  function openDashboard() {
    document.getElementById('granted-screen').style.display = 'none';
    document.getElementById('dashboard-screen').style.display = 'block';
  }
  
  function goBack() {
    document.getElementById('denied-screen').style.display = 'none';
    document.getElementById('welcome-screen').style.display = 'block';
  }



const camera = document.getElementById('camera');

navigator.mediaDevices
  .getUserMedia({ video: true }) // Request video access
  .then((stream) => {
    // Set the video stream to the video element
    camera.srcObject = stream;
  })
  .catch((error) => {
    console.error("Error accessing the camera: ", error);
    alert("Unable to access the camera. Please check your permissions.");
  });
  
  // Dataset of Users
const users = [
    {
        name: "Michael Wijaya",
        age: 21,
        role: "Staff",
        photo: "images/michael.jpg", // Replace with actual image path
    },
    {
        name: "John Doe",
        age: 30,
        role: "Manager",
        photo: "images/john.jpg",
    },
];

// Open and Close Modal
const modal = document.getElementById("access-modal");
const openModalBtn = document.querySelector(".open-modal-btn");
const closeModalBtn = document.querySelector(".close-btn");

// Function to Display User Info
function displayUser(name) {
    const user = users.find((user) => user.name === name);

    if (user) {
        document.getElementById("user-name").textContent = user.name;
        document.getElementById("user-age").textContent = user.age;
        document.getElementById("user-role").textContent = user.role;
        document.getElementById("user-photo").src = user.photo;
    } else {
        alert("User not found!");
        modal.style.display = "none"; // Close modal if user is not found
    }
}

// Function to Simulate User Detection
function simulateDetection() {
    const detectedUserName = "Michael Wijaya"; // Simulated detected name
    displayUser(detectedUserName);
    modal.style.display = "flex"; // Show the modal
}

// Function to Proceed (when ENTER is clicked)
function proceed() {
    alert("Proceeding to the next step...");
    // Add navigation logic here (e.g., window.location.href = "nextpage.html";)
}

// Event Listeners
openModalBtn.addEventListener("click", simulateDetection);
closeModalBtn.addEventListener("click", () => (modal.style.display = "none"));

// Close Modal on Outside Click
window.addEventListener("click", (event) => {
    if (event.target === modal) {
        modal.style.display = "none";
    }
});

