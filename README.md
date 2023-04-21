The solution for both tasks is displayed in the main.ipynb file. For both tasks, the points are obtained by using OpenCV. Based on those retrieved points in combination with the given points, image objects are created, which inherent functions that are the basis for calibrating and reconstructing.
The notebook can be executed without selecting the points again, as the resulting files are already provided in the output folder. All results are already provided in the output folder without needing to execute any code. Executing the code will overwrite the files in the output folder

## Calibration

For the first tasks, direct linear calibration is performed to calculate the projection matrix. Next, the intrinsic and extrinsic parameters are obtained. 


## Reconstruction

For the second task, the 3D coordinates for each point pair is calculated by utilizing the projection matrix. The quality of the estimated points is evaluated with the mse-methode. The obtained value of around 5 seem acceptable based on the results, but could be improved upon, when using a computer mouse, iPad or manually adjusting the pixel coordinates in the txt-files. The results are saved and can be directly plotted by just executing the last cell of the notebook after importing numpy and matplotlib.pyplot.