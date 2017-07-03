# Realtime Object Recognition Skill

A real-time object recognition skill for Mycroft AI using [Google's TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection) and [OpenCV](http://opencv.org/).

This skill is a proof of concept to use tensorflow and openCV to provide realtime object recognition using the default webcam as a source.  So far it's only been tested on Ubuntu, and since it's resource heavy may be laggy or not work on less powerful machines running Mycroft.

Hopefully this is just a start and with optimization and further development this skill will provide a concept to create more skills around object recognition

## Requirements
- [Mycroft](https://docs.mycroft.ai/installing.and.running/installation)
- [TensorFlow 1.2](https://www.tensorflow.org/)
- [OpenCV 3.0](http://opencv.org/)

## Installation

Clone the repository into your skills directory. Then install the
dependencies inside your mycroft virtual environment:

If on picroft just skip the workon part and the directory will be /opt/mycroft/skills

```
cd /opt/mycroft/skills (or wherever your working skills directory is located)
git clone https://github.com/eClarity/skill-realtime-object-recognition
workon mycroft
cd skill-realtime-object-recognition
pip install -r requirements.txt
```

## Usage:
* `View Objects`

