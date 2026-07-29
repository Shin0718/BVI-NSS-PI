# Blind & Visually Impaired Navigation Simulation System (BVI-NSS)

BVI-NSS is a local simulation tool for studying non-visual navigation by blind and visually impaired (BVI) users. The system combines a route environment, risk inference, attention gating, and ACT-R based action selection to examine how device feedback, environmental references, and cognitive load may shape walking behavior.

Current status: **Beta · Research Prototype**. The current web designer UI and backend parameter-mapping layer are connected for internal research testing.

This repository is intentionally limited to the runnable system. Calibration materials, generated figures, cached map files, and previous simulation outputs are not included.

## Quick Start

Clone the repository:

```bash
git clone git@github.com:Shin0718/BVI-SAS.git
cd BVI-SAS
```

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the local web interface:

```bash
python server.py
```

Open the local site in a browser:

```text
http://127.0.0.1:8766
```

The site runs on the user's own machine. No hosted server, account system, or cloud deployment is required.

## Repository Structure

```text
BVI-SAS/        Core simulation modules
web/            Local browser interface
reports/        Empty output directory for generated reports
server.py       Local web server and simulation API
requirements.txt
```

Main modules:

- `main.py`: command-line entry point for single-run and Monte Carlo simulations.
- `simulation.py`: simulation loop and ACT-R interaction logic.
- `environment.py`: map and route environment construction.
- `inference.py`: risk inference utilities.
- `actr_setup.py`: ACT-R model, buffers, chunks, and production setup.
- `reporting.py`: CSV, JSON, Markdown, and figure output generation.

## Requirements

The code was developed for Python 3.12 or later. The required Python packages are listed in `requirements.txt`:

- `matplotlib`
- `networkx`
- `numpy`
- `osmnx`
- `pyactr`

## Local Web Interface

The local browser interface allows designers or researchers to configure route-template intervention rules and user/context controls before running a simulation.

The current designer page includes:

- Product functions: obstacle detection, terrain detection, pedestrian detection, vehicle approach detection, guidance object detection, and other information detection.
- Trigger conditions: obstacle, vehicle, pedestrian, landmark, surface change, and none.
- Route-template slots: segment slots and intersection slots. Segment slots map to road-segment rules; the intersection slot maps only to crossing/intersection rules.
- Device parameters: recognition strength, trigger probability, false/missed alert probability, active duration, preparation duration, feedback modality, alert duration, alert frequency, and competition factor.
- Guidance-object targets: curb, wall, railing, tactile paving, and landmark.

Start the local server from the repository root:

```bash
python server.py
```

Then open:

```text
http://127.0.0.1:8766
```

The web interface sends the selected configuration to the local `/simulate` endpoint. The server maps the route-template slots, selected functions/triggers, user profile, scenario controls, and device parameters to runtime model parameters before each simulation run. Overrides are applied only for the active run and are restored afterward, so the source defaults in `BVI-SAS/` remain unchanged.

The backend mapping layer currently connects the designer UI to the existing simulation system. It does not replace the core ACT-R simulation engine.

Generated reports are written to `reports/`. Each UI-triggered run records both simulation outputs and the frontend input/mapping context:

- `sim_summary_<timestamp>.json`
- `sim_ui_config_<timestamp>.json`
- `sim_io_report_<timestamp>.md`

Runtime outputs, map caches, and Python cache files are ignored by git.

## Backend Mapping Status

The current frontend-to-backend mapping is implemented and ready for beta testing:

- Segment slots apply to segment-level route rules across the simulated route.
- Intersection slots apply to crossing/intersection rules only.
- Device trust maps to `DEVICE_TRUST` and trust-related manual confirmation burden.
- Familiarity maps to the existing familiarity condition used by the ACT-R simulation.
- Obstacle detection maps to obstacle and small-obstacle recognition parameters.
- Pedestrian detection maps to pedestrian activity trigger probabilities.
- Vehicle approach detection maps to segment or crossing vehicle-approach probabilities depending on slot scope.
- Terrain detection maps to surface-type probability distribution parameters.
- Guidance object detection maps selected targets to curb, wall, railing, tactile paving, and landmark-related parameters.
- Feedback modality, alert duration, alert frequency, and competition factor affect channel load and attention-competition terms.

## Running the Simulation

The simulation engine can also be run directly from the command line.

Run a single simulation:

```bash
python BVI-SAS/main.py
```

Run with a familiarity condition:

```bash
python BVI-SAS/main.py --familiarity 0
python BVI-SAS/main.py --familiarity 1
```

Run multiple simulations:

```bash
python BVI-SAS/main.py --familiarity 1 --mc-runs 50 --seed-start 20260701
```

Generated files are written to `reports/`.

## Notes

The folder name `BVI-SAS` follows the project naming used for release. Because the hyphen is not a valid Python package character, the recommended entry point is direct script execution:

```bash
python BVI-SAS/main.py
```

## Citation

If this code is used in academic work, please cite the associated manuscript when available.
