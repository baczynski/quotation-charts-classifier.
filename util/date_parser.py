import datetime
import time


def date_to_timestamp(date, date_format_regular_expression):
    return time.mktime(datetime.datetime.strptime(date, date_format_regular_expression).timetuple())
