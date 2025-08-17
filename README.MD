
# Elyx Member Journey Dashboard  

This project solves the given problem statement:  

> **Generate communication messages over 8 months from member sign up, visualize the member’s journey, and provide reasoning for Elyx’s decisions.**  

The app simulates WhatsApp-style conversations between a **member** and the **Elyx team** (doctor, coach, nutritionist, etc.), then converts them into structured JSON and visualizes the **member’s progress, decisions, and internal metrics**.  

---

## 📦 Requirements  

- **Python 3.8+**  
- **Node.js 16+**  
- **npm 8+**  

---

## 🐍 Python Setup  

It’s recommended to use a **virtual environment**:  

```bash
python -m venv venv
````

Activate:

* Mac/Linux: `source venv/bin/activate`
* Windows: `.\venv\Scripts\activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

To generate `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

## 🚀 Workflow

1. **Generate Conversations (8 months)**

   ```bash
   python agent_new.py
   ```

   This creates `.txt` files simulating conversations based on constraints:

   * ✅ 1 diagnostic test every 3 months
   * ✅ \~5 member-initiated queries per week
   * ✅ Weekly reports from Elyx
   * ✅ Exercise updated every 2 weeks
   * ✅ Travel once every 4 weeks
   * ✅ 50% adherence to plans (leading to adjustments)

2. **Convert Conversations to JSON**

   ```bash
   python conv_to_json.py
   ```

   This produces structured JSON with fields like:

   ```json
   {
     "date": "2025-01-15",
     "sender": "Doctor",
     "role": "Medical",
     "message": "We recommend a diagnostic panel this week to track your blood sugar.",
     "tags": ["diagnostic", "blood_marker"]
   }
   ```

3. **Run the React Dashboard**

   ```bash
   npm install
   npm start
   ```

   Access at 👉 [http://localhost:3000](http://localhost:3000)

4. **Build for Production**

   ```bash
   npm run build
   ```

---

## 📂 Project Structure

```
elyx-dashboard/
├── public/              # Static assets
├── src/                 # React components & pages
├── agent_new.py         # Generates conversations (follows PS rules)
├── conv_to_json.py      # Converts conversations to JSON
├── tailwind.config.js   # TailwindCSS config
├── postcss.config.js    # PostCSS config
├── package.json         # React dependencies & scripts
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## 🛠️ Tech Stack

* **Python** – conversation generation + preprocessing
* **React + TailwindCSS** – frontend visualization
* **PostCSS** – CSS transformations

---

## 📊 Dashboard Features

* **Timeline View** → WhatsApp-style messages day by day
* **Analytics Dashboard** →

  * Hours spent by Elyx doctors/coaches
  * Symptom progress over time
  * Diagnostic test results every 3 months
* **Decision Traceback** → See *why* a treatment or plan was recommended (e.g., high blood sugar → diet change)
* **Persona Analysis** → Member traits, adherence patterns, travel disruptions

---

## ⚡ Quick Start (One-liner)

```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python agent_new.py && python conv_to_json.py && npm install && npm start
```

---

✅ After following these steps, you’ll get a **simulated 8-month journey**, fully visualized in the React dashboard.


