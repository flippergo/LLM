# LLM

This repository provides a few Python utilities including an object detection example.

## Download `object_detection.py`

You can grab the object detection script without cloning the entire repository:

```bash
curl -L -o object_detection.py https://raw.githubusercontent.com/flippergo/LLM/main/object_detection.py
```

Make it executable and launch the Streamlit app:

```bash
chmod +x object_detection.py
streamlit run object_detection.py
```

The web interface lets you upload an image, set a confidence threshold with a
slider, view detection results and optionally download the annotated output.
