*** Settings ***
Library    main.py


*** Variables ***
${model_name}    ${CURDIR}\\model.keras

${training_path}    ${CURDIR}\\Data\\training\\
${validation_path}    ${CURDIR}\\Data\\validation\\


${test_phtot_1}        ${CURDIR}\\test_left.jpg
${test_phtot_2}        ${CURDIR}\\test_left2.jpg
${test_phtot_3}        ${CURDIR}\\test_right.jpg
${test_phtot_4}        ${CURDIR}\\test_st.jpg


*** Test Cases ***

Training a model
    Train Model    ${training_path}    ${validation_path}



detect photo
    Predict From Path    ${model_name}        ${test_phtot_4}


capture a photo 
    Capture And Predict    ${model_name}

