# Data format
for every AV
input:[ID,frame,x,y,left_front_x,left_front_y,left_behind_x,left_behind_y,front_x,front_y,behind_x,behind_y,right_front_x,right_front_y,right_behind_x,right_behind_y] of previous 80 frame(0.1 s)
output:[x, y] in the next frame(0.1 s)

# Pub
@article{ZHANG2019287,
title = "Simultaneous modeling of car-following and lane-changing behaviors using deep learning",
journal = "Transportation Research Part C: Emerging Technologies",
volume = "104",
pages = "287 - 304",
year = "2019",
issn = "0968-090X",
doi = "https://doi.org/10.1016/j.trc.2019.05.021",
url = "http://www.sciencedirect.com/science/article/pii/S0968090X18308003",
author = "Xiaohui Zhang and Jie Sun and Xiao Qi and Jian Sun",
keywords = "Traffic flow, Car-following, Lane-changing, Simultaneous modeling, Deep learning, Long Short-Term Memory",
abstract = "Car-following (CF) and lane-changing (LC) behaviors are two basic movements in traffic flow which are generally modeled separately in the literature, and thus the interaction between the two behaviors may be easily ignored in separated models and lead to unrealistic traffic flow description. In this paper, we adopt a deep learning model, long short-term memory (LSTM) neural networks, to model the two basic behaviors simultaneously. By only observing the position information of the six vehicles surrounding the subject vehicle, the LSTM can extract the significant features that influence the CF and LC behaviors automatically and predict the vehicles behaviors with time-series data and memory effects. In addition, we propose a hybrid retraining constrained (HRC) training method to further optimize the LSTM model. With the I-80 trajectory data of NGSIM dataset we train and test the HRC LSTM model, while the results show that HRC LSTM model can accurately estimate CF and LC behaviors simultaneously with low longitudinal trajectories error and high LC prediction accuracy compared with the classical models. We also evaluate the transferability of the proposed model with the US101 dataset and a good transferability result is obtained as well."
}
