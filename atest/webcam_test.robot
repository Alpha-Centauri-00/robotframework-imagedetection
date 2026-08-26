*** Settings ***
Documentation    Optional webcam smoke test - separate from smoke_test.robot because it
...              needs a physical camera and something in frame to point it at, so it
...              can't run unattended. Run smoke_test.robot first to get a trained
...              model, then point ${MODEL_PATH} below at that model's filename before
...              running this suite.
...
...              The prediction itself won't mean much (the model only knows the
...              synthetic Left/Right classes) - this is just to confirm the camera
...              capture -> crop -> predict -> log.html pipeline runs without error.
...
...              If your machine has more than one camera, override ${CAMERA_INDEX}
...              instead of editing this file - e.g. in a local robot.toml:
...                  [variables]
...                  CAMERA_INDEX = "1"
...              or on the command line with `robot --variable CAMERA_INDEX:1 ...`.
Library          Imagedetection

*** Variables ***
${MODEL_PATH}       model_202626081215.pt
${CAMERA_INDEX}     0

*** Test Cases ***
Capture From Webcam And Predict
    [Documentation]    Activates the webcam, captures a photo, and runs prediction.
    ...    Check log.html afterwards to see the captured photo and result.
    ${d}    Detect From Webcam    ${MODEL_PATH}    camera_index=${CAMERA_INDEX}
