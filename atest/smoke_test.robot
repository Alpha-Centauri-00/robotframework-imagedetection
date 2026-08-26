*** Settings ***
Documentation    End-to-end smoke test for the Imagedetection library: confirms which
...              device (CPU or CUDA) is in use, then runs a full train -> save ->
...              load -> predict cycle against a small synthetic dataset.
...
...              This only proves the pipeline works and exercises the GPU if present -
...              it is not meant to prove real-world detection accuracy. Run
...              atest/generate_sample_data.py once before this suite.
Library          Imagedetection

*** Variables ***
${TRAINING_DIR}       ${CURDIR}/Data/training
${VALIDATION_DIR}     ${CURDIR}/Data/validation
${RED_SAMPLE1}        ${CURDIR}/Data/test_photos/red_sample1.png
${RED_SAMPLE2}        ${CURDIR}/Data/test_photos/red_sample2.png
${BLUE_SAMPLE1}       ${CURDIR}/Data/test_photos/blue_sample1.png
${BLUE_SAMPLE2}       ${CURDIR}/Data/test_photos/blue_sample2.png
${GREEN_SAMPLE1}      ${CURDIR}/Data/test_photos/green_sample1.png
${GREEN_SAMPLE2}      ${CURDIR}/Data/test_photos/green_sample2.png
${MODEL_PATH}         ${CURDIR}

*** Test Cases ***
Report Device Info
    [Documentation]    Logs PyTorch/CUDA version and GPU name (if any) to log.html.
    ...    If you installed a CUDA-enabled PyTorch build and have an NVIDIA GPU,
    ...    this should report "cuda" - if it reports "cpu" instead, PyTorch fell
    ...    back to CPU (check your torch install, see README's GPU support section).
    ${device}=    Get Device Info
    Log To Console    \nDevice in use: ${device}

Train A Small Model
    [Documentation]    Trains on the synthetic Red/Blue/Green dataset. Also the
    ...    main keyword to watch for GPU usage (check nvidia-smi / Task Manager's
    ...    GPU graph while this runs).
    ${model}=    Train Model    ${TRAINING_DIR}    ${VALIDATION_DIR}
    Should Not Be Empty    ${model}
    Set Suite Variable    ${MODEL_PATH}    ${model}

Detect Red Sample1 From Path
    ${prediction}=    Detect From Path    ${MODEL_PATH}    ${RED_SAMPLE1}
    Should Be Equal As Strings    ${prediction}    Red

Detect Red Sample2 From Path
    ${prediction}=    Detect From Path    ${MODEL_PATH}    ${RED_SAMPLE2}
    Should Be Equal As Strings    ${prediction}    Red

Detect Blue Sample1 From Path
    ${prediction}=    Detect From Path    ${MODEL_PATH}    ${BLUE_SAMPLE1}
    Should Be Equal As Strings    ${prediction}    Blue

Detect Blue Sample2 From Path
    ${prediction}=    Detect From Path    ${MODEL_PATH}    ${BLUE_SAMPLE2}
    Should Be Equal As Strings    ${prediction}    Blue

Detect Green Sample1 From Path
    ${prediction}=    Detect From Path    ${MODEL_PATH}    ${GREEN_SAMPLE1}
    Should Be Equal As Strings    ${prediction}    Green

Detect Green Sample2 From Path
    ${prediction}=    Detect From Path    ${MODEL_PATH}    ${GREEN_SAMPLE2}
    Should Be Equal As Strings    ${prediction}    Green
