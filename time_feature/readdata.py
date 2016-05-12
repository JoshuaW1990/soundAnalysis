import numpy as np
import xlrd
import csv
from math import floor, ceil, log, exp


xl_wb = xlrd.open_workbook("set01.xlsx")

# Read the data
sheet_names = xl_wb.sheet_names()

input_set = [] # list of data
output_set = [] # 1 ~ 9
for i in range(1, len(sheet_names)):
    sheet_name = sheet_names[i]
    xl_ws = xl_wb.sheet_by_name(sheet_name)
    for j in range(xl_ws.ncols):
        instance = []
        if j % 2 == 0:
            continue
        column = xl_ws.col(j)
        for k in range(len(column)):
            if k <= 404:
                continue
            cell = column[k]
            value = cell.value
            instance.append(value)
        input_set.append(instance)
        output_set.append(i)

# Handling continuous value
flatten_input = []
for instance in input_set:
    flatten_input += instance
max_value = ceil(max(flatten_input))
min_value = floor(min(flatten_input))
gap = max_value - min_value
gap_fold_num = 10
gap_fold_size = float(gap) / float(gap_fold_num)
preprocess_input = []
for instance in input_set:
    preprocess_instance = []
    for value in instance:
        tmp_value = value - min_value
        value = float(floor(float(tmp_value) / gap_fold_size))
        preprocess_instance.append(value)
    preprocess_input.append(preprocess_instance)

col_name = [str(item) for item in range(len(preprocess_input[0]))] + ["label"]

with open("preprocess.csv", 'wb') as csvfile:
    file = csv.writer(csvfile)
    col_name = [str(item) for item in range(len(preprocess_input[0]))] + ["label"]
    file.writerow(col_name)
    for i in range(len(preprocess_input)):
        label = output_set[i]
        instance = preprocess_input[i]
        data = instance + [label]
        file.writerow(data)





