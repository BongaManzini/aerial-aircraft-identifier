# ️ AeroScan AI: Deep Learning Aerial Image Aircraft Identifier

> **UniSA Annual Hackathon Challenge Project** 
> **Challenge:** "Given these images, use AI and ML techniques to identify aircraft." 
> **Deliverable:** A platform where the user drops an image of an aircraft and it is correctly detected, classified, and analyzed. 
> **Author:** Bonga Manzini (UNISA BSc Informatics Student)

---

## The Story Behind This Project

At the **UniSA Annual Hackathon**, we were given aerial and satellite imagery of airfields and tasked with building a computer vision model that can detect and classify aircraft.

Identifying aircraft from the sky is completely different from ordinary photos:
1. **No Fixed Up/Down Orientation:** An aircraft on an apron or taxiway can face any rotation angle from $0^\circ$ to $360^\circ$.
2. **Harsh Tarmac Glare & Shadows:** Sun shadows can sometimes be larger than the plane itself, and white runway stripes can fool standard edge detectors.
3. **Scale Variance:** A light Cessna 172 is tiny compared to a four-engine C-130 Hercules transport or a Boeing 777.

I built **AeroScan AI** as an interactive platform where you can drag and drop an aerial photo (or choose from preset flightline samples) and immediately see the detected bounding box, classified aircraft category, and aerodynamic telemetry.

---

## Aircraft Taxonomy & Classes

AeroScan AI classifies aircraft into 6 operational categories:
- **Military Fighter Jet:** Delta or cropped swept wings with internal fuselage turbofans (e.g., JAS 39 Gripen, F-16 Falcon).
- **Commercial Jetliner:** Low swept wings with dual under-wing podded turbofans (e.g., Boeing 737, Airbus A320).
- **Heavy Cargo Transport:** High straight wings with four turboprop/turbofan nacelles (e.g., Lockheed C-130 Hercules).
- **Rotary Wing / Helicopter:** Central main rotor disc and tail boom assembly (e.g., Denel Rooivalk, Bell 206, Oryx).
- **General Aviation Propeller:** High/low straight wings with single nose piston engine (e.g., Cessna 172 Skyhawk).
- **Tactical UAV Drone:** High aspect-ratio glider wings with rear pusher prop and twin boom (e.g., Bayraktar TB2).

---

## ️ How It Works (Technical Pipeline)

1. **Preprocessing & Normalization:** Converts imagery to 224x224 RGB, applying **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to normalize runway glare.
2. **Deep Learning Backbone:** Utilizes a lightweight convolutional backbone (MobileNetV2 / ResNet) optimized for fast edge inference.
3. **Rotational Invariant Augmentation:** Trained with random 360-degree rotation and affine transforms to handle aircraft at any heading.
4. **Interactive Browser Platform:** Runs in real-time in any web browser without requiring a heavy GPU backend server.

---

## Running the Project Locally

### 1. Interactive Web Platform
Open `index.html` in your web browser, or serve it with Python:
```bash
python3 -m http.server 8080
# Open http://localhost:8080 in your browser
```

### 2. Python CLI Inference Pipeline
```bash
python3 aircraft_classifier.py sample_gripen_fighter.jpg
```

---

## Contributing & Pull Requests

Fellow students and computer vision enthusiasts are welcome to contribute!
- Adding more aircraft classes (e.g., business jets, seaplanes, supersonic transports).
- Improving YOLO or ONNX model weights.
- Enhancing satellite imagery tile scrapers.

Feel free to fork, make your changes, and open a Pull Request!
