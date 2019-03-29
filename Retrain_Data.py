import numpy as np


def retrain(x_test_id, x_test, model):
    single_test = x_test[0]
    all_predicted = []
    all_re_input = []
    all_re_input.append(single_test)
    cons_num = 0
    for i in range(len(x_test) - 1):
        pre_y = single_test[-1][1]
        predicted = model.predict(np.reshape(single_test, (1, 80, 14)))
        all_predicted.append(predicted)
        if x_test_id[i + 1][0, 0] == x_test_id[i][0, 0] and x_test_id[i + 1][0, 1] - x_test_id[i][0, 1] == 1:
            single_test = np.delete(single_test, 0, 0)
            last_row = x_test[i + 1][-1]
            # safety constrains-time headway > 1s
            if predicted[0, 1] > last_row[9] and last_row[7] - predicted[0, 1] > 1 * (
                    predicted[0, 1] - pre_y) * 10:  # range_y >= 0.2 and range_y <= 0.8:
                # last_row[0] = predicted[0,0]
                last_row[1] = predicted[0, 1]
            cons_num += 1
            single_test = np.row_stack((single_test, last_row))
            # single_test[-1][2] = predicted[0,0]
            # single_test[-1][3] = predicted[0,1]
        else:
            single_test = x_test[i + 1]
        all_re_input.append(single_test)
    last_predicted = model.predict(np.reshape(single_test, (1, 80, 14)))
    all_predicted.append(last_predicted)

    return all_re_input, all_predicted, cons_num / len(x_test)
