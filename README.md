##MOCPIL in Matlab

This is the code for our paper “Privileged Multi-view One-class Support Vector Machine”, 2023.

1. Introduction
In this paper, a novel multi-view one-class support vector machine method with privileged information learning (MOCPIL) is put forward. MOCPIL embodies both the consensus principle and complementarity principle in multi-view learning. Privileged information is additional data that is available only in the training process, but not in the testing process. By introducing the idea of privileged information learning, MOCPIL implements the complementarity principle by treating one view as the training data and the other view as the privileged data. Moreover, MOCPIL implements the consensus principle by requiring that different views of the same object should give similar predicting outputs. The learning problem of MOCPIL is a quadratic programming (QP) problem, which is able to be solved by off-the-shelf QP solvers.

2. Requirements
Matlab R2021b version

3. Datasets
MOCPIL is evaluated on several multi-view one-class classification datasets, i.e., Handwritten, Caltech-7 and NUS-WIDE. 

(2) For the above datasets, the data format is as follows: 
- Handwritten: Pix(240), Fou(76)
- Caltech-7: Gabor(48), WM(40)
- NUS-WIDE: CH(65), CM(226)

(3) All the above experimental datasets can be downloaded from link https://mega.nz/folder/O7wzXR6L#-Hisbcv_Hw-1_aVPkmJITQ. 

4. Run codes
(1)How to run the codes?
1. Download dataset via https://mega.nz/folder/O7wzXR6L#-Hisbcv_Hw-1_aVPkmJITQ.
2. Run main.m directly.

5. Output
The average AUC and standard deviation.

For further any inquires, feel free to contact me at pguiting@163.com or post an issue here.


