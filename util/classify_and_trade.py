import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_yaml
from pyspark import SparkConf, SparkContext

from masterthesis.nowe.MAIN import load_model
from masterthesis.util.date_parser import date_to_timestamp


def classify_and_trade(file_name, number_of_attributes, timestamp_index, bid_index, ask_index,
                       separator, quotation_size, quotation_step, date_format_regular_expression):
    [win, loss] = [0, 0]
    [quotation_timestamps, bid_quotes, ask_quotes] = load_file(file_name, number_of_attributes, timestamp_index,
                                                               bid_index, ask_index, separator,
                                                               date_format_regular_expression)

    model = load_model()

    for i in range(0, len(quotation_timestamps), quotation_step):
        timestamps_batch = quotation_timestamps[i:(i + quotation_size)]
        timestamps_batch[:] = [x / 10000000000 for x in timestamps_batch]
        prices_batch = ask_quotes[i: (i + quotation_size)]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.axis('off')
        ax.plot(timestamps_batch, prices_batch)
        fig.savefig('./image.png')

        image = cv2.imread('./image.png', cv2.IMREAD_GRAYSCALE)
        (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        x = np.empty((1, 480, 640), dtype=np.float32)
        x[0, ...] = im_bw

        c = model.predict_classes(x)

        # if c == 1:
        #     copyfile('./image.png', '../predicted_real_time/1/' + str(i) + '.png')
        # elif c == 2:
        #     copyfile('./image.png', '../predicted_real_time/2/' + str(i) + '.png')
        # elif c == 3:
        #     copyfile('./image.png', '../predicted_real_time/3/' + str(i) + '.png')
        # elif c == 4:
        #     copyfile('./image.png', '../predicted_real_time/4/' + str(i) + '.png')

        if c != 0:
            growing_trend = check_growing_trend(bid_quotes, i, quotation_step)
            [win, loss] = trade(c, bid_quotes, ask_quotes, i + quotation_size, win, loss, growing_trend)


def load_file(file_name, number_of_attributes, timestamp_index, bid_index, ask_index, separator,
              date_format_regular_expression):
    conf = SparkConf().setMaster("local").setAppName("My App")
    SparkContext._ensure_initialized()
    sc = SparkContext(conf=conf)
    textRDD = sc.textFile(file_name)

    quotations = textRDD.flatMap(lambda x: x.split(separator)).zipWithIndex() \
        .filter(
        lambda q: q[1] % number_of_attributes == timestamp_index or q[1] % number_of_attributes == bid_index or q[
            1] % number_of_attributes == ask_index)

    quotation_timestamps = quotations.filter(lambda q: q[1] % number_of_attributes == timestamp_index) \
        .map(
        lambda timestamp: date_to_timestamp(timestamp[0], date_format_regular_expression)) \
        .collect()

    bid_quotes = quotations.filter(lambda q: q[1] % number_of_attributes == bid_index) \
        .map(lambda timestamp: float(timestamp[0])) \
        .collect()

    ask_quotes = quotations.filter(lambda q: q[1] % number_of_attributes == ask_index) \
        .map(lambda timestamp: float(timestamp[0])) \
        .collect()

    return [quotation_timestamps, bid_quotes, ask_quotes]


def load_model():
    yaml_file = open('../model/model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("../model/model.h5")
    print("Loaded model from disk")
    return loaded_model


#
def trade(class_number, bid_quotes, ask_quotes, quotation_index, win, loss, growing_trend):
    if class_number == 1:
        if not growing_trend:
            [win, loss] = sell_and_watch(bid_quotes, ask_quotes, quotation_index, take_profit=1, stop_loss=1,
                                         percent=True, win=win, loss=loss)
    if class_number == 2:
        if growing_trend:
            [win, loss] = buy_and_watch(bid_quotes, ask_quotes, quotation_index, take_profit=1, stop_loss=1,
                                        percent=True, win=win, loss=loss)
    if class_number == 3:
        if growing_trend:
            [win, loss] = buy_and_watch(bid_quotes, ask_quotes, quotation_index, take_profit=1, stop_loss=1,
                                        percent=True, win=win, loss=loss)
    if class_number == 4:
        if growing_trend:
            [win, loss] = sell_and_watch(bid_quotes, ask_quotes, quotation_index, take_profit=1, stop_loss=1,
                                         percent=True, win=win, loss=loss)

    print('current win: ' + str(win))
    print('current loss: ' + str(loss))
    return [win, loss]


def buy_and_watch(bid_quotes, ask_quotes, quotation_index, take_profit, stop_loss, percent, win, loss):
    trade_price = ask_quotes[quotation_index]
    take_profit_value = 0
    stop_loss_value = 0

    current_bid_price = bid_quotes[quotation_index]
    index = quotation_index

    if percent:
        take_profit_value = current_bid_price * (100 + take_profit) / 100
        stop_loss_value = current_bid_price * (100 - stop_loss) / 100
    else:
        take_profit_value = current_bid_price + take_profit
        stop_loss_value = current_bid_price - stop_loss

    while stop_loss_value < current_bid_price < take_profit_value and index + 1 < len(bid_quotes):
        index = index + 1
        current_bid_price = bid_quotes[index]

    if index != len(bid_quotes):
        if current_bid_price >= take_profit_value:
            return [win + 1, loss]
        else:
            return [win, loss + 1]
    else:
        return [win, loss]


def sell_and_watch(bid_quotes, ask_quotes, quotation_index, take_profit, stop_loss, percent, win, loss):
    trade_price = bid_quotes[quotation_index]
    take_profit_value = 0
    stop_loss_value = 0

    current_bid_price = ask_quotes[quotation_index]
    index = quotation_index

    if percent:
        take_profit_value = current_bid_price * (100 - take_profit) / 100
        stop_loss_value = current_bid_price * (100 + stop_loss) / 100
    else:
        take_profit_value = current_bid_price - take_profit
        stop_loss_value = current_bid_price + stop_loss

    while take_profit_value < current_bid_price < stop_loss_value and index + 1 < len(bid_quotes):
        index = index + 1
        current_bid_price = ask_quotes[index]

    if index != len(ask_quotes):
        if current_bid_price <= take_profit_value:
            return [win + 1, loss]
        else:
            return [win, loss + 1]
    else:
        return [win, loss]


def check_growing_trend(bid_quotes, index, quotation_step):
    if index - quotation_step >= 0:
        if bid_quotes[index] >= bid_quotes[index - quotation_step]:
            return True
        else:
            return False
    else:
        return True


classify_and_trade('../../real_time/quotations/USDJPY_2017_15M.csv', 6, 0, 3, 4, ',', 76, 10, "%d.%m.%Y %H:%M:%S.%f")