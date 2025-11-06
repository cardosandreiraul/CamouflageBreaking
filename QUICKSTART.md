# Quick Start Guide

## For Colleagues: Get Running in 5 Minutes

### 1. Clone the repo
```bash
git clone https://github.com/Raul-Andrei-Cardos/CamouflageBreaking.git
cd CamouflageBreaking
```

### 2. Set up Python environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get the dataset
- Download the "Camo Animals" dataset
- Place it in `data/Camo Animals/`
- Or update `DATASET_PATH` in `main.py` to point to your dataset location

### 5. Run it!
```bash
python main.py
```

That's it! You should see the algorithm analysis running.