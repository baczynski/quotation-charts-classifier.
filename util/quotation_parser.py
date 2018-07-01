import matplotlib.pyplot as plt
from pyspark import SparkConf, SparkContext

from masterthesis.util.date_parser import date_to_timestamp


def parse_file(input_file_name, output_file_name, number_of_attributes, timestamp_index, price_index,
               separator, quotation_size, quotation_step, date_format_regular_expression):
    conf = SparkConf().setMaster("local").setAppName("My App")
    SparkContext._ensure_initialized()
    sc = SparkContext(conf=conf)
    textRDD = sc.textFile(input_file_name)

    quotations = textRDD.flatMap(lambda x: x.split(separator)).zipWithIndex() \
        .filter(
        lambda q: q[1] % number_of_attributes == timestamp_index or q[1] % number_of_attributes == price_index)

    quotation_timestamps = quotations.filter(lambda q: q[1] % number_of_attributes == timestamp_index) \
        .map(
        lambda timestamp: date_to_timestamp(timestamp[0], date_format_regular_expression)) \
        .collect()

    quotation_prices = quotations.filter(lambda q: q[1] % number_of_attributes == price_index) \
        .map(lambda timestamp: float(timestamp[0])) \
        .collect()

    for i in range(0, len(quotation_timestamps), quotation_step):
        timestamps_batch = quotation_timestamps[i:(i + quotation_size)]
        timestamps_batch[:] = [x / 10000000000 for x in timestamps_batch]
        prices_batch = quotation_prices[i: (i + quotation_size)]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.axis('off')
        ax.plot(timestamps_batch, prices_batch)
        fig.savefig(output_file_name + 'USDJPY15-' + str(int(i/10)) + '.png')


parse_file('../../real_time/USDJPY_2017_15M.csv', '../../real_time/images/', 6, 0, 4, ',', 76, 10, "%d.%m.%Y %H:%M:%S.%f")
