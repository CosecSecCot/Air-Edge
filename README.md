# Air Edge

Air Edge is a virtual air hockey game that uses **computer vision (OpenCV)** to
track a physical mallet while simulating the puck in **Pygame**. It also supports
**multiplayer over WebSockets**.

## Project Setup

### **1. Clone the Repository**

```sh
git clone https://github.com/your-repo/air-edge.git
cd air-edge
```

### **2. Install Poetry (if not installed)**

Poetry is used to manage dependencies. Install it with:

```sh
pip install poetry
```

### **3 Configure Poetry to Create Virtual Environments Inside the Project**

```sh
poetry config virtualenvs.in-project true
```

### **4. Create & Activate Virtual Environment**

```sh
poetry install
```

This installs all dependencies inside a virtual environment.

#### **Activate the Virtual Environment (if needed)**

- **Linux/macOS:**  

  ```sh
  source $(poetry env info --path)/bin/activate
  ```

- **Windows (PowerShell):**  

  ```powershell
  & (poetry env info --path)\Scripts\activate
  ```

### **5. Run the Project**

Start the game:

```sh
poetry run python src/main.py
```

### **6. Deactivating the Virtual Environment**

To exit the virtual environment:

```sh
deactivate
```

---

## Issues & Support

If you face any issues, feel free to
[open an issue](https://github.com/your-repo/air-edge/issues) or reach out!
