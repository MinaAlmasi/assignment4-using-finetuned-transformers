# activate virtual environment 
source ./env/bin/activate

# run classification
echo -e "[INFO:] Running Emotion Classification using BERT..."
python src/classify_emotion.py

# run visualisations
echo -e "[INFO:] Creating Visualizations ..."
python src/visualise_emotion.py

# deactivate env 
deactivate

# happy user msg!! 
echo -e "[INFO:] Pipeline complete! Visualisations stored"