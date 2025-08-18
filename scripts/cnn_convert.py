import qkeras as qk
from qkeras import *
import layers as fql

import pandas as pd
import numpy as np
import os

from tqdm import tqdm
import traceback
import matplotlib.pyplot as plt

import pathlib


DATA_WIDTH = 16
FWL = 10
INT_BITS = DATA_WIDTH - FWL
PATH_OUT_LOC = os.path.join("data", "scratchpad") + os.sep


class ArgsClass:
    verbosity = True


args = ArgsClass
debug = True
if debug:
    args.verbosity = True

values_as_index_filt = False
values_as_index_img = False


def dict_to_list_two_tupel(dictionary):
    item_list = dictionary.items()
    result_list = list()
    for elem in item_list:
        result_list.append((elem[0], elem[1][0], elem[1][1]))
    return result_list


def bindigits(n, bits, signed: bool):
    # clip if signed, overflow if unsigned
    if signed:
        if n > 2**bits - 1:
            print("Warning: signed clipping")
            print(traceback.format_stack()[-2])
        n = min(2 ** (bits - 1) - 1, n)
        n = max(-(2 ** (bits - 1)), n)
    else:
        if n > 2**bits - 1:
            print("Warning: unsigned overflow")
            print(traceback.format_stack()[-2])
        # n = 2 ** bits - 1
        n = max(0, n)
    s = bin(n & int("1" * bits, 2))[2:]
    return ("{0:0>%s}" % bits).format(s)


def invert_bindigits(n, bits):
    n_bin = bindigits(n, bits, False)
    if n_bin[0] == "1":
        n_bin_inv = "".join("1" if x == "0" else "0" for x in n_bin)
        result = -(int(n_bin_inv, 2) + 1)
    else:
        result = n
    return result


def calc_filt_mem(
    msg_mem_size_list_local,
    msg_flat_list_local,
    max_rand_int_loc,
    filt_weights_loc=None,
    bias_weights_loc=None,
):
    mem_filt_msg = list()
    flat_filt_msg_local = list()

    for elem in msg_mem_size_list_local:
        if elem[0] == "k" or elem[0] == "s" or elem[0] == "r" or elem[0] == "c":
            mem_filt_msg.append(elem)
    mem_filt_msg = sorted(mem_filt_msg, key=lambda tup: tup[1])

    for elem in msg_flat_list_local:
        if elem[0] == "k" or elem[0] == "s" or elem[0] == "r" or elem[0] == "c":
            flat_filt_msg_local.append(elem)
    flat_filt_msg_local = sorted(flat_filt_msg_local, key=lambda tup: tup[1])

    if debug:
        print("filter mem and flat message")
        print(mem_filt_msg)
        print(flat_filt_msg_local)

    # create filter and bias mem image
    bias_mem_local = list()
    filt_mem_local = list()

    # check dimensions, of external filter and bias data
    if not (filt_weights_loc is None):
        if filt_weights_loc.shape != (
            mem_filt_msg[1][2],
            mem_filt_msg[0][2],
            mem_filt_msg[2][2],
            mem_filt_msg[3][2],
        ):
            print("ERROR: Filter data does not match the layer dimensions")
            print("Mem filt dimensions:\t" + str(mem_filt_msg))
            print("External filt value dimensions:\t" + str(filt_weights_loc.shape))
            sys.exit("ERROR: Filter data does not match the layer dimensions")
    if not (bias_weights_loc is None):
        if bias_weights_loc.shape[0] != mem_filt_msg[3][2]:
            print("ERROR: Bias data does not match the layer dimensions")
            print("Mem bias dimensions:\t" + str(mem_filt_msg[3][2]))
            print("External bias value dimensions:\t" + str(bias_weights_loc.shape))
            # sys.exit()

    for ind_3 in range(mem_filt_msg[3][2]):  # k
        if not (bias_weights_loc is None):
            bias_mem_local.append(bias_weights_loc[ind_3])
        elif values_as_index_filt:
            bias_mem_local.append(ind_3 + 1)
        else:
            bias_mem_local.append(int(max_rand_int_loc * np.random.rand()))
        for ind_2 in range(mem_filt_msg[2][2]):  # c
            for ind_1 in range(mem_filt_msg[1][2]):  # r
                for ind_0 in range(mem_filt_msg[0][2]):  # s
                    if not (filt_weights_loc is None):
                        filt_mem_local.append(filt_weights_loc[ind_1][ind_0][ind_2][ind_3])
                    elif values_as_index_filt:
                        filt_mem_local.append(
                            ind_0
                            + ind_1 * np.prod([mem_filt_msg[i][2] for i in range(1)])
                            + ind_2 * np.prod([mem_filt_msg[i][2] for i in range(2)])
                            + ind_3 * np.prod([mem_filt_msg[i][2] for i in range(3)])
                        )
                    else:
                        filt_mem_local.append(
                            int(max_rand_int_loc * np.random.rand() - max_rand_int_loc / 2)
                        )

    mem_filt_offset_local = dict()
    for j, elem in enumerate(mem_filt_msg):
        mem_filt_offset_local[elem[0]] = int(np.prod([mem_filt_msg[i][2] for i in range(j)]))

    buf_filt_offset_local = dict()
    for j, elem in enumerate(flat_filt_msg_local):
        buf_filt_offset_local[elem[0]] = int(np.prod([flat_filt_msg_local[i][2] for i in range(j)]))

    return filt_mem_local, mem_filt_offset_local, buf_filt_offset_local, bias_mem_local


def calc_img_mem(
    msg_mem_size_list_local,
    msg_flat_list_local,
    max_rand_int_loc,
    stride_x_loc,
    stride_y_loc,
    img_values_loc=None,
):
    mem_img_msg_local = list()
    flat_img_msg_local = list()
    flat_result_msg_local = list()

    delta_s = 0
    delta_r = 0

    for elem in msg_mem_size_list_local:
        if elem[0] == "q":
            # tmp_elem = (elem[0], elem[1], elem[2] + delta_s)
            # tmp_elem = (elem[0], elem[1], int(np.ceil(elem[2] / stride_x_loc)))
            flat_result_msg_local.append(elem)
        if elem[0] == "w":
            # tmp_elem = (elem[0], elem[1], elem[2] + delta_s)
            mem_img_msg_local.append(elem)
            flat_img_msg_local.append(elem)
        if elem[0] == "p":
            # tmp_elem = (elem[0], elem[1], elem[2] + delta_r)
            # tmp_elem = (elem[0], elem[1], int(np.ceil(elem[2] / stride_y_loc)))
            flat_result_msg_local.append(elem)
        if elem[0] == "h":
            # tmp_elem = (elem[0], elem[1], elem[2] + delta_r)
            mem_img_msg_local.append(elem)
            flat_img_msg_local.append(elem)
        if elem[0] == "c":
            mem_img_msg_local.append(elem)
            flat_img_msg_local.append(elem)
            # TODO: das sollte doch eigentlich k sein, weil result, oder?
            flat_result_msg_local.append(elem)

    mem_img_msg_local = sorted(mem_img_msg_local, key=lambda tup: tup[1])
    flat_img_msg_local = sorted(flat_img_msg_local, key=lambda tup: tup[1])
    flat_result_msg_local = sorted(flat_result_msg_local, key=lambda tup: tup[1])

    # flat_result_msg_local['p'][2] = int(np.ceil(flat_result_msg_local['p'][2] / stride_x_loc))

    if debug:
        print("image mem and flat and result flat message")
        print(mem_img_msg_local)
        print(flat_img_msg_local)
        print(flat_result_msg_local)

    # check dimensions, of external filter and bias data
    if not (img_values_loc is None):
        if img_values_loc.shape != (
            mem_img_msg_local[2][2],
            mem_img_msg_local[1][2],
            mem_img_msg_local[0][2],
        ):
            print("ERROR: img data does not match the layer dimensions")
            print("Mem img dimensions:\t" + str(mem_img_msg_local))
            print("External img value dimensions:\t" + str(img_values_loc.shape))
            # sys.exit()

    img_mem_local = list()

    for ind_2 in range(mem_img_msg_local[2][2]):  # c
        for ind_1 in range(mem_img_msg_local[1][2]):  # p
            for ind_0 in range(mem_img_msg_local[0][2]):  # q
                if not (img_values_loc is None):
                    tmp_val = int(img_values_loc[ind_2][ind_1][ind_0])
                    # print('index c, p, q: {0}, {1}, {2}; value: {3}'.format(ind_2, ind_1, ind_0, tmp_val))
                    img_mem_local.append(tmp_val)
                elif values_as_index_img:
                    img_mem_local.append(
                        ind_0
                        + ind_1 * np.prod([mem_img_msg_local[i][2] for i in range(1)])
                        + ind_2 * np.prod([mem_img_msg_local[i][2] for i in range(2)])
                    )
                else:
                    img_mem_local.append(
                        int(max_rand_int_loc * np.random.rand() - max_rand_int_loc / 2)
                    )

    mem_img_offset_local = dict()
    for j, elem in enumerate(mem_img_msg_local):
        mem_img_offset_local[elem[0]] = int(np.prod([mem_img_msg_local[i][2] for i in range(j)]))

    buf_img_offset_local = dict()
    for j, elem in enumerate(flat_img_msg_local):
        buf_img_offset_local[elem[0]] = int(np.prod([flat_img_msg_local[i][2] for i in range(j)]))

    # TODO: Account for Stride
    buf_result_offset_local = dict()
    for j, elem in enumerate(flat_result_msg_local):
        buf_result_offset_local[elem[0]] = int(
            np.prod([flat_result_msg_local[i][2] for i in range(j)])
        )

    if debug:
        print("result offset")
        print(buf_result_offset_local)

    return (
        img_mem_local,
        mem_img_offset_local,
        buf_img_offset_local,
        buf_result_offset_local,
    )


def write_buffer(mem_image, file_name, buf_depth, data_with_loc):
    textfile = open(file_name + ".txt", "w")
    textfile_bin = open(file_name + "_bin.txt", "w")

    textfile.write("Buffer Depth in Words:\t" + str(buf_depth) + "\n")
    textfile_bin.write("{0:016b}".format(buf_depth) + "\n")

    for j, data in enumerate(mem_image):
        textfile.write(str(j) + ", " + str(data) + "\n")
        textfile_bin.write("{0}".format(bindigits(int(data), data_with_loc, True)) + "\n")

    textfile.close()
    textfile_bin.close()


def write_calc(file_name, calc_len, filt_calc_loc, img_calc_loc, accum_calc_loc, res_addr_calc_loc):
    textfile = open(file_name + ".txt", "w")

    textfile.write("Buffer Depth in Words:\t" + str(calc_len) + "\n")

    for j in range(calc_len):
        textfile.write(
            "cycle: {3}; res_addr: {4}; filt: {0}; img: {1}; accum: {2}\n".format(
                filt_calc_loc[j],
                img_calc_loc[j],
                accum_calc_loc[j],
                j,
                res_addr_calc_loc[j],
            )
        )

    textfile.close()


def write_layer_config_file(
    file_name,
    msg_flat_dict_loc,
    buf_img_offset_loc,
    buf_filt_offset_loc,
    buf_result_offset_loc,
    img_buf_depth,
    filt_buf_depth,
    result_buf_depth,
    stride_x_loc,
    stride_y_loc,
    bias_buf_len_loc,
    bias_loc,
    relu_loc,
    fwl_loc,
):
    dimensions_dict = {
        "layer_conf": "111",
        "k": "110",
        "s": "001",
        "r": "010",
        "c": "011",
        "q": "100",
        "p": "101",
    }
    print(buf_result_offset_loc)
    bias_state = ["bias off", "bias on"]
    relu_state = ["relu off", "relu on"]
    stride_x_log2 = int(np.log2(stride_x_loc))
    stride_y_log2 = int(np.log2(stride_y_loc))
    bin_sep_str = ""
    textfile = open(file_name + ".txt", "w")
    textfile_bin = open(file_name + "_bin.txt", "w")

    textfile.write("Dim - mem size - offset - split/accum index (q)\n")

    filt_buf_depth_lower = filt_buf_depth & 65535
    filt_buf_depth_upper = filt_buf_depth - filt_buf_depth_lower >> 16
    print(filt_buf_depth)
    print(filt_buf_depth_lower)
    print(filt_buf_depth_upper)

    for i, cur_dim in enumerate(dimensions_dict):
        # dimension
        # the first layer conf word is currently written directly by the sequence, thus it must be omitted in the conf file
        if cur_dim != "layer_conf":
            textfile.write(cur_dim + ": " + str(msg_flat_dict_loc[cur_dim][1]))
            textfile_bin.write(
                dimensions_dict[cur_dim]
                + bin_sep_str
                + "{0:013b}".format(msg_flat_dict_loc[cur_dim][1])
                + "\n"
            )
        # buf offset message
        if cur_dim == "p":
            textfile.write(", " + str(buf_img_offset_loc["w"]))
            textfile_bin.write("{0:016b}".format(buf_img_offset_loc["w"]) + bin_sep_str)
        if cur_dim == "q":
            textfile.write("\nmem offset of p result - bias buf len\n")
            # The result offset is written here in purpose, because the img q offset is always 1
            # and the result q offset is not trivial derived from the img offset due to the stride
            textfile.write(str(buf_result_offset_loc["p"]))
            textfile_bin.write("{0:012b}".format(buf_result_offset_loc["p"]))
            textfile.write(", " + str(0) + "\n")
            textfile_bin.write("{0:04b}\n".format(0) + bin_sep_str)
            # append the result buffer after q, to save a state in the reading fsm in LINA
            textfile.write("Result Buffer Depth in Words:\t" + str(0) + "\n")
            textfile_bin.write("{0:016b}".format(0) + "\n")
            textfile.write("Layer config:\t" + bias_state[bias_loc])
            textfile_bin.write("{0:01b}".format(bias_loc) + bin_sep_str)
            textfile.write(", " + relu_state[relu_loc])
            textfile_bin.write("{0:01b}".format(relu_loc) + bin_sep_str)
            textfile.write(", stride x = " + str(stride_x_log2))
            textfile_bin.write("{0:02b}".format(stride_x_log2) + bin_sep_str)
            textfile.write(", stride y = " + str(stride_y_log2))
            textfile_bin.write("{0:02b}".format(stride_y_log2) + bin_sep_str)
            textfile.write(", fwl: " + str(fwl_loc))
            textfile_bin.write("{0:010b}".format(fwl_loc) + bin_sep_str)
        if cur_dim == "s" or cur_dim == "r":
            textfile.write(", " + str(buf_filt_offset_loc[cur_dim]))
            textfile_bin.write("{0:016b}".format(buf_filt_offset_loc[cur_dim]) + bin_sep_str)
        # split message
        if cur_dim == "k":
            textfile.write(", " + str(buf_filt_offset_loc[cur_dim]))
            textfile_bin.write("{0:012b}".format(buf_filt_offset_loc[cur_dim]) + bin_sep_str)
            textfile.write(", " + str(0))
            textfile_bin.write("{0:04b}".format(0) + bin_sep_str)
        if cur_dim == "c":
            textfile.write("\nbuf img offset - buf filt offset - split index - buf result offset\n")
            textfile.write(str(buf_img_offset_loc[cur_dim]))
            textfile_bin.write("{0:016b}".format(buf_img_offset_loc[cur_dim]) + "\n")
            textfile.write(", " + str(buf_filt_offset_loc[cur_dim]))
            textfile_bin.write("{0:012b}".format(buf_filt_offset_loc[cur_dim]) + bin_sep_str)
            textfile.write(", " + str(0))
            textfile_bin.write("{0:04b}".format(0) + "\n")
            textfile.write(", " + str(buf_result_offset_loc[cur_dim]))
            textfile_bin.write("{0:016b}".format(buf_result_offset_loc[cur_dim]))
        if cur_dim != "layer_conf":
            textfile.write("\n")
            textfile_bin.write("\n")

    textfile.write("Filt Buffer Depth in Words Lower:\t" + str(filt_buf_depth_lower) + "\n")
    textfile_bin.write("{0:016b}\n".format(filt_buf_depth_lower))
    textfile.write("Filt Buffer Depth in Words Upper:\t" + str(filt_buf_depth_upper))
    textfile_bin.write("{0:016b}".format(filt_buf_depth_upper) + "\n")

    textfile.close()
    textfile_bin.close()


def translate_layer(
    lay_num_loc,
    lay_inf_loc,
    img_float_loc,
    filt_float_loc,
    bias_float_loc,
    fwl_loc,
    data_width_loc,
):

    # to have enough channels for splitting in the next layer, filter kernels must be great enough
    layer_name = "l{0}_usecase".format(lay_num_loc)
    print(layer_name)
    stride_x = lay_inf_loc.loc[lay_num_loc, "Ux"]
    stride_y = lay_inf_loc.loc[lay_num_loc, "Uy"]
    msg_mem_dict_l0 = {
        "k": (3, lay_inf_loc.loc[lay_num_loc, "M"]),
        "s": (0, lay_inf_loc.loc[lay_num_loc, "S"]),
        "r": (1, lay_inf_loc.loc[lay_num_loc, "R"]),
        "c": (2, lay_inf_loc.loc[lay_num_loc, "C"]),
        "w": (0, lay_inf_loc.loc[lay_num_loc, "W"]),
        "h": (1, lay_inf_loc.loc[lay_num_loc, "H"]),
        "q": (0, lay_inf_loc.loc[lay_num_loc, "E"]),
        "p": (1, lay_inf_loc.loc[lay_num_loc, "F"]),
    }
    print(msg_mem_dict_l0)
    print("")

    ref_out = np.zeros((msg_mem_dict_l0["k"][1], msg_mem_dict_l0["p"][1], msg_mem_dict_l0["q"][1]))
    # ref_out = np.zeros((6, 1, 4000))
    ref_out.dtype = np.dtype(np.int64)
    print(ref_out.shape)
    print(msg_mem_dict_l0)

    MAX_POS_VAL = 2 ** (data_width_loc - 1) - 1
    MAX_NEG_VAL = -(2 ** (data_width_loc - 1))

    filt_calc = list()
    img_calc = list()
    accum_calc = list()
    res_addr_calc = list()

    for ind_5 in tqdm(range(msg_mem_dict_l0["k"][1])):
        for ind_4 in range(msg_mem_dict_l0["p"][1]):
            ind_4_stride = ind_4 * stride_y
            for ind_3 in range(msg_mem_dict_l0["q"][1]):
                ind_3_stride = ind_3 * stride_x
                ref_out[ind_5, ind_4, ind_3] = bias_float_loc[ind_5]
                for ind_2 in range(msg_mem_dict_l0["c"][1]):
                    for ind_1 in range(msg_mem_dict_l0["r"][1]):
                        for ind_0 in range(msg_mem_dict_l0["s"][1]):
                            # calc w and h
                            ind_w = ind_3_stride + ind_0
                            ind_h = ind_4_stride + ind_1
                            filt_tmp = filt_float_loc[ind_0, ind_1, ind_2, ind_5]
                            try:
                                img_tmp = img_float_loc[ind_2, ind_h, ind_w]
                            except IndexError:
                                print(
                                    "IndexError at ind_2, ind_h, ind_w: {0}, {1}, {2}".format(
                                        ind_2, ind_h, ind_w
                                    )
                                )
                                raise
                            ref_out[ind_5, ind_4, ind_3] = (
                                filt_tmp * img_tmp + ref_out[ind_5, ind_4, ind_3]
                            )
                            filt_calc.append(filt_tmp)
                            img_calc.append(img_tmp)
                            accum_calc.append(ref_out[ind_5, ind_4, ind_3])
                            res_addr_calc.append(
                                str(
                                    ind_3
                                    + ind_4 * msg_mem_dict_l0["q"][1]
                                    + ind_5 * msg_mem_dict_l0["q"][1] * msg_mem_dict_l0["p"][1]
                                )
                            )
                # ref_out[ind_5, ind_4_stride, ind_3_stride] = max(min(int(tmp >> FWL), MAX_POS_VAL), MAX_NEG_VAL)
                if ref_out[ind_5][ind_4][ind_3] < 0 and relu == 1:
                    ref_out[ind_5][ind_4][ind_3] = 0
                tmp = int(ref_out[ind_5][ind_4][ind_3]) >> (fwl_loc + 0)
                ref_out[ind_5][ind_4][ind_3] = max(min(tmp, MAX_POS_VAL), MAX_NEG_VAL)

    max_rand_int = 2 ** (fwl_loc - 0)

    msg_flat_dict = dict()
    msg_flat_dict["k"] = msg_mem_dict_l0["k"]
    msg_flat_dict["c"] = msg_mem_dict_l0["c"]
    msg_flat_dict["s"] = msg_mem_dict_l0["s"]
    msg_flat_dict["r"] = msg_mem_dict_l0["r"]
    msg_flat_dict["q"] = msg_mem_dict_l0["w"]
    msg_flat_dict["p"] = msg_mem_dict_l0["h"]

    # dim, pos, size
    msg_flat_list = dict_to_list_two_tupel(msg_flat_dict)
    msg_mem_size_list = dict_to_list_two_tupel(msg_mem_dict_l0)

    if debug:
        print("Flat dict/Temporal calc order:\t" + str(msg_flat_dict))

    # create filter and image mem file
    filt_float_loc = filt_float_loc.transpose((1, 0, 2, 3))
    filt_mem, mem_filt_offset, buf_filt_offset, bias_mem = calc_filt_mem(
        msg_mem_size_list, msg_flat_list, max_rand_int, filt_float_loc, bias_float_loc
    )

    bias_mem_shift = list()

    for elem in bias_mem:
        bias_mem_shift.append(int(elem * 2 ** -(fwl_loc)))

    filt_bias_mem = bias_mem_shift + filt_mem
    img_mem, mem_img_offset, buf_img_offset, buf_result_offset = calc_img_mem(
        msg_mem_size_list,
        msg_flat_list,
        max_rand_int,
        stride_x,
        stride_y,
        img_values_loc=img_float_loc,
    )

    result_mem = list()

    for ind_2 in range(msg_mem_dict_l0["k"][1]):  # c
        for ind_1 in range(msg_mem_dict_l0["p"][1]):  # p
            for ind_0 in range(msg_mem_dict_l0["q"][1]):  # q
                tmp_val = int(ref_out[ind_2][ind_1][ind_0])
                # print('index c, p, q: {0}, {1}, {2}; value: {3}'.format(ind_2, ind_1, ind_0, tmp_val))
                result_mem.append(tmp_val)

    layer_name_loc = "l{0}_waveform".format(lay_num_loc)

    if False:
        print("Filt mem:\t" + str(filt_mem))
        print("Bias mem:\t" + str(bias_mem))
        print("Img mem:\t" + str(img_mem))

    file_name = "{0}calc_".format(PATH_OUT_LOC)
    # write_calc(file_name + layer_name_loc, len(filt_calc), filt_calc, img_calc, accum_calc, res_addr_calc)

    # write buffer content to file
    file_name = "{0}filt_buf_".format(PATH_OUT_LOC)
    write_buffer(filt_bias_mem, file_name + layer_name_loc, len(filt_bias_mem), data_width_loc)
    file_name = "{0}img_buf_".format(PATH_OUT_LOC)
    write_buffer(img_mem, file_name + layer_name_loc, len(img_mem), data_width_loc)
    file_name = "{0}result_buf_".format(PATH_OUT_LOC)
    write_buffer(result_mem, file_name + layer_name_loc, len(result_mem), data_width_loc)

    # file_name = '{4}calc_order_{0}_k{1}_c{2}_{3}'.format(pe_num_layer, msg_split_list_loc[0][2], msg_split_list_loc[1][2], layer_name_loc, path_out_loc)

    # write layer config file
    file_name = "{0}config_{1}".format(PATH_OUT_LOC, layer_name_loc)
    # Account for stride
    msg_flat_dict["q"] = (
        msg_flat_dict["q"][0],
        int(np.floor(msg_flat_dict["q"][1] / stride_x)),
    )
    msg_flat_dict["p"] = (
        msg_flat_dict["p"][0],
        int(np.floor(msg_flat_dict["p"][1] / stride_y)),
    )
    if debug:
        print(msg_flat_dict)
    # TODO: write config file
    print(buf_result_offset)
    write_layer_config_file(
        file_name,
        msg_flat_dict,
        buf_img_offset,
        buf_filt_offset,
        buf_result_offset,
        len(img_mem),
        len(filt_bias_mem),
        ref_out.size,
        stride_x,
        stride_y,
        bias_buf_len_loc=len(bias_mem),
        bias_loc=1,
        relu_loc=1,
        fwl_loc=fwl_loc,
    )

    # calculate naive reference
    # ref_out = calc_naive_reference(msg_mem_dict_loc, mem_filt_offset, mem_img_offset, filt_mem, img_mem, bias_mem, stride_x, stride_y)

    # apply relu
    # calc_out_relu, calc_out_uvm_relu, ref_out_relu = apply_relu(msg_mem_dict_loc, ref_out, calc_out, calc_out_uvm, relu_loc, fwl_loc, data_width_loc, stride_x, stride_y)

    # write result to file, for external comparison with uvm result
    # file_name = '{4}result_buffer_{0}_k{1}_c{2}_{3}'.format(pe_num_layer, msg_split_list_loc[0][2], msg_split_list_loc[1][2], layer_name_loc, path_out_loc)
    # write_result_buffer(calc_out_uvm_relu, file_name, 2 ** msg_split_list_loc[1][2], pe_num_layer)

    return ref_out


def transpose_flatten_weights(lay_num_loc, lay_inf_loc, layer_weights_loc):
    filt_mem_ext_loc = np.zeros(
        (1, 1, lay_inf_loc.loc[lay_num_loc, "C"], lay_inf_loc.loc[lay_num_loc, "M"])
    )
    print("Flatten Neurons:\t" + str(lay_inf_loc.loc[lay_num_loc, "C"]))

    for index_k_next in range(lay_inf_loc.loc[lay_num_loc, "M"]):
        for index_c_loc in range(lay_inf_loc.loc[lay_num_loc - 1, "M"]):
            for index_h_loc in range(lay_inf_loc.loc[lay_num_loc - 1, "F"]):
                for index_w_loc in range(lay_inf_loc.loc[lay_num_loc - 1, "E"]):
                    index_c_next_lina = (
                        index_w_loc
                        + index_h_loc * lay_inf_loc.loc[lay_num_loc - 1, "E"]
                        + index_c_loc
                        * lay_inf_loc.loc[lay_num_loc - 1, "E"]
                        * lay_inf_loc.loc[lay_num_loc - 1, "F"]
                    )
                    index_c_next_tf = (
                        index_c_loc
                        + index_w_loc * lay_inf_loc.loc[lay_num_loc - 1, "M"]
                        + index_h_loc
                        * lay_inf_loc.loc[lay_num_loc - 1, "M"]
                        * lay_inf_loc.loc[lay_num_loc - 1, "E"]
                    )
                    filt_mem_ext_loc[0][0][index_c_next_lina][index_k_next] = layer_weights_loc[
                        lay_num_loc
                    ][index_c_next_tf][index_k_next]

    filt_mem_ext_loc = filt_mem_ext_loc
    return filt_mem_ext_loc


if len(sys.argv) < 2:
    print("Usage: python translate_cnn.py <model_path>")
    sys.exit(1)


# load model
# model_path = '/home/klein/git/ha_cnn/output/Output/models/MNIST_min_cnn_q3_16_fully_quant'
model_path = sys.argv[1]
model_qkeras = tf.keras.models.load_model(
    model_path,
    custom_objects={
        "FullyQConv2D": fql.FullyQConv2D,
        "QActivation": qk.QActivation,
        "FullyQDense": fql.FullyQDense,
        "quantized_relu": qk.quantized_relu,
        "FullyQAveragePooling2D": fql.FullyQAveragePooling2D,
    },
)


# Conversion to 16 Bit
layer_weights = list()
layer_bias = list()
layer_weights_float = list()
layer_bias_float = list()
layer_type = list()

d = {
    "type": [],  # Convolution or fully connected
    "config": [],  # flatten or other status bits for the DLA
    "activation": [],  # relu or no relu
    "H": [],  # Height of layer input
    "W": [],  # Width of layer input
    "S": [],  # Width of filter kernel
    "R": [],  # Height of filter kernel
    "Ux": [],  # Stride for kernel in x-direction
    "Uy": [],  # Stride for kernel in y-direction
    "C": [],  # Number of input channels
    "M": [],  # Number of filters(conv)/neurons(fully connected)
    "F": [],  # Height of layer output
    "E": [],  # Width of layer output
    "MP_x": [],  # Height of maxpooling window
    "MP_y": [],  # Width of maxpooling window
    "MP_Ux": [],  # Stride for maxpooling in x-direction
    "MP_Uy": [],  # Stride for maxpooling in x-direction
    "img_size": [],
    "bias_size": [],
    "filt_size": [],
    "psum_size": [],
    "img_buf_depth": [],
    "filt_buf_depth": [],
    "cycles": [],
    "corresponding_layer": [],
}
lay_inf = pd.DataFrame(data=d)
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Dataframe Erzeugung zur besseren Übersicht

layers = model_qkeras.layers
lay_name_conv = "FullyQConv2D"
lay_name_dense = "FullyQDense"
layer_name_relu = "relu"
# layers_q = model_q.layers

for i in range(len(layers)):
    layer = layers[i]
    lay_name_tmp = layer.__class__.__name__
    if lay_name_tmp == lay_name_dense or lay_name_tmp == lay_name_conv:
        weights = qk.quantized_bits(DATA_WIDTH, INT_BITS, 1)(layer.kernel)
        layer_weights.append((weights * 2 ** (FWL - 0)).numpy().astype(np.int32))
        layer_weights_float.append(weights.numpy())
        # layer_weights_float.append(layer.kernel.numpy())

        bias = qk.quantized_bits(DATA_WIDTH, INT_BITS, 1)(layer.bias)
        layer_bias.append((bias * 2 ** (2 * FWL - 0)).numpy().astype(np.int32))
        layer_bias_float.append(bias.numpy())
        # layer_bias_float.append(layer.bias.numpy())

        lay_inf.loc[i, "corresponding_layer"] = i
        # Must be set to 1, in order to keep posterior calculations working
        lay_inf.loc[i, "MP_x"] = 1
        lay_inf.loc[i, "MP_y"] = 1
        lay_inf.loc[i, "MP_Ux"] = 1
        lay_inf.loc[i, "MP_Uy"] = 1
        lay_inf.loc[i, "config"] = "none"
        # lay_inf.loc[i, "activation"] = str(layer.activation)[10:14]
        lay_inf.loc[i, "activation"] = ""

        if i < (len(layers) - 1):
            # check if string 'max_pool' is contained in layer name
            next_1 = layers[min(i + 1, len(layers) - 1)].name.find("max_pool") >= 0
            next_2 = layers[min(i + 2, len(layers) - 1)].name.find("max_pool") >= 0
            if (next_1 or next_2) and layers[i + 1].name.find("conv") < 0:
                max_pool_index = [next_1, next_2].index(True) + 1
                # Note order of rows and columns
                lay_inf.loc[i, "MP_x"] = layers[i + max_pool_index].pool_size[1]
                lay_inf.loc[i, "MP_y"] = layers[i + max_pool_index].pool_size[0]
                lay_inf.loc[i, "MP_Ux"] = layers[i + max_pool_index].strides[1]
                lay_inf.loc[i, "MP_Uy"] = layers[i + max_pool_index].strides[0]
                lay_inf.loc[i, "config"] = "max_pool"
            if layers[i + 1].__class__.__name__ == "QActivation":
                # check if string 'relu' is contained in layer name
                if layers[i + 1].__name__.find(layer_name_relu) >= 0:
                    lay_inf.loc[i, "activation"] = "relu"
        if i > 0:
            if layers[i - 1].__class__.__name__ == "Flatten":
                lay_inf.loc[i, "config"] = "flatten"
        lay_inf.loc[i, "type"] = lay_name_tmp
        if lay_name_tmp == lay_name_conv:
            # tf order is: Cols first, Rows second
            # Note: Height(H) is Number of Rows
            lay_inf.loc[i, "H"] = layer.input_shape[2]
            lay_inf.loc[i, "W"] = layer.input_shape[1]
            lay_inf.loc[i, "C"] = layer.input_shape[3]
            lay_inf.loc[i, "M"] = layer.kernel.shape[3]
            # Same goes for Filter Dims
            lay_inf.loc[i, "S"] = layer.kernel.shape[0]
            lay_inf.loc[i, "R"] = layer.kernel.shape[1]
            # Stride should also be considered
            lay_inf.loc[i, "Ux"] = layer.strides[0]
            lay_inf.loc[i, "Uy"] = layer.strides[1]
            # Account for Stride
            lay_inf.loc[i, "F"] = int(
                np.floor(
                    (lay_inf.loc[i, "H"] - lay_inf.loc[i, "R"] + lay_inf.loc[i, "Uy"])
                    / lay_inf.loc[i, "Uy"]
                )
            )
            # Account for Max Pool
            # lay_inf.loc[i, 'E'] = (lay_inf.loc[i, 'E'] - lay_inf.loc[i, 'MP_x'] + lay_inf.loc[i, 'MP_Ux']) / lay_inf.loc[i, 'MP_Ux']
            # Account for Stride
            lay_inf.loc[i, "E"] = int(
                np.floor(
                    (lay_inf.loc[i, "W"] - lay_inf.loc[i, "S"] + lay_inf.loc[i, "Ux"])
                    / lay_inf.loc[i, "Ux"]
                )
            )
            # Account for Max Pool
            # lay_inf.loc[i, 'F'] = (lay_inf.loc[i, 'F'] - lay_inf.loc[i, 'MP_y'] + lay_inf.loc[i, 'MP_Uy']) / lay_inf.loc[i, 'MP_Uy']
        elif lay_name_tmp == lay_name_dense:
            lay_inf.loc[i, "H"] = 1
            lay_inf.loc[i, "W"] = 1
            lay_inf.loc[i, "C"] = layer.input_shape[1]
            lay_inf.loc[i, "M"] = layer.kernel.shape[1]
            # Same goes for Filter Dims
            lay_inf.loc[i, "R"] = 1
            lay_inf.loc[i, "S"] = 1
            lay_inf.loc[i, "F"] = 1
            lay_inf.loc[i, "E"] = 1
            lay_inf.loc[i, "Ux"] = 1
            lay_inf.loc[i, "Uy"] = 1
        print("Layer ", layer.name, " converted")
    else:
        print("Warning: Layer ", layer.name, " unknown.")

number_layers = lay_inf.shape[0]

lay_inf = lay_inf.convert_dtypes()
lay_inf = lay_inf.set_index(pd.Index(range(number_layers)))

img_start = 0
max_img = 0
max_psum = 0
sum_bias_size = 0
sum_filt_size = 0
max_psum_lay_index = 0

# Bestimmung der Größe von img, filt, psum und bias, sowie der jeweiligen Maxima
for i in range(number_layers):
    lay_inf.loc[i, "img_size"] = lay_inf.loc[i, "H"] * lay_inf.loc[i, "W"] * lay_inf.loc[i, "C"]
    lay_inf.loc[i, "bias_size"] = lay_inf.loc[i, "M"]
    if lay_inf.loc[i, "config"] == "flatten":
        lay_inf.loc[i, "filt_size"] = (
            lay_inf.loc[i, "M"]
            * lay_inf.loc[i, "R"]
            * lay_inf.loc[i, "S"]
            * lay_inf.loc[i, "H"]
            * lay_inf.loc[i, "W"]
            * lay_inf.loc[i, "C"]
        )
    else:
        lay_inf.loc[i, "filt_size"] = (
            lay_inf.loc[i, "M"] * lay_inf.loc[i, "R"] * lay_inf.loc[i, "S"] * lay_inf.loc[i, "C"]
        )
    lay_inf.loc[i, "psum_size"] = lay_inf.loc[i, "M"] * lay_inf.loc[i, "F"] * lay_inf.loc[i, "E"]
    sum_bias_size += lay_inf.loc[i, "bias_size"]
    sum_filt_size += lay_inf.loc[i, "filt_size"]
    # TODO: Check, if this works for more complicated examples
    if np.mod(i, 2) == 0:
        if max_psum < lay_inf.loc[i, "psum_size"]:
            max_psum_lay_index = i
        max_img = np.max([max_img, lay_inf.loc[i, "img_size"]])
        max_psum = np.max([max_psum, lay_inf.loc[i, "psum_size"]])
        pass
    elif np.mod(i, 2) == 1:
        if max_psum < lay_inf.loc[i, "img_size"]:
            max_psum_lay_index = i
        max_psum = np.max([max_psum, lay_inf.loc[i, "img_size"]])
        max_img = np.max([max_img, lay_inf.loc[i, "psum_size"]])

print(lay_inf)

lina_ref_result = list()
lina_float_result = list()

bias_config = 1
relu = 1
number_layers = lay_inf.shape[0]


def get_2d_quantized(waveform, bitwidth, integer_bits):
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    waveform = waveform[..., tf.newaxis]
    waveform = qk.quantized_bits(bitwidth, integer_bits - 1, 0)(waveform)
    return waveform


def make_2d_ds_quantized(ds, bitwidth, integer_bits):
    return ds.map(
        map_func=lambda audio, label: (
            get_2d_quantized(audio, bitwidth, integer_bits),
            label,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


# file_name = './' + model_path + '_sample.npy'
# sample_in = np.load(file_name)
# sample_in = np.expand_dims(sample_in, 0)

DATASET_PATH = "data/mini_speech_commands"

data_dir = pathlib.Path(DATASET_PATH)
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset="both",
)

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

test_2d_ds_q = make_2d_ds_quantized(test_ds, DATA_WIDTH, INT_BITS)

# choose test audio sample out of 64 batch
test_index = 5

for example_audio, example_labels in test_2d_ds_q.take(1):
    print(example_audio.shape)
    print(example_labels.shape)

sample_in = example_audio.numpy()[test_index, :, :, :]
sample_out = example_labels.numpy()[test_index]

# transfer tensorflow image to lina
img_in_float = sample_in
img_in = (img_in_float * 2 ** (FWL + 0)).astype(np.int32)

# TODO: Last index should be 0, shouldn't it?
img_in_shuffle = np.zeros((img_in.shape[2], img_in.shape[1], img_in.shape[0]))
img_in_shuffle_float = np.zeros((img_in.shape[2], img_in.shape[1], img_in.shape[0]))

print(img_in_shuffle.shape)
print(img_in.shape)

for index_c in range(img_in.shape[2]):
    for index_h in range(img_in.shape[1]):
        for index_w in range(img_in.shape[0]):
            img_in_shuffle[index_c][index_h][index_w] = img_in[index_w][index_h][index_c]
            img_in_shuffle_float[index_c][index_h][index_w] = img_in_float[index_w][index_h][
                index_c
            ]

ref_out_l_next = img_in_shuffle.astype(np.int32)
# ref_out_l_next_float = img_in_shuffle_float #.astype(np.int32)


img_org = np.expand_dims(sample_in, axis=0)

print(img_org.shape)

img_uvm = None

# convert CNN: conv - flatten - fully connected
for l_index in range(number_layers):
    if lay_inf.loc[l_index, "config"] == "flatten":
        filt_mem_ext = transpose_flatten_weights(l_index, lay_inf, layer_weights).astype(np.int32)
        print(l_index)
        print(ref_out_l_next.shape)
        ref_out_l_next = ref_out_l_next.reshape((lay_inf.loc[l_index, "C"], 1, 1))
    else:
        if lay_inf.loc[l_index, "type"] == lay_name_dense:
            filt_mem_ext = layer_weights[l_index].reshape(
                1, 1, lay_inf.loc[l_index, "C"], lay_inf.loc[l_index, "M"]
            )
        else:
            filt_mem_ext = layer_weights[l_index]
    bias_mem_ext = layer_bias[l_index]
    ref_out_l_next = translate_layer(
        l_index, lay_inf, ref_out_l_next, filt_mem_ext, bias_mem_ext, FWL, DATA_WIDTH
    )
    lina_float_result.append(ref_out_l_next)

print("\n\nfinished use case cnn\n\n")

# Print Total Mem Usage and Total Cylces

print("Total cycles:\t", lay_inf.loc[:, "cycles"].sum())
print("Total filt buffer size:\t", lay_inf.loc[:, "filt_buf_depth"].sum())
print("Max img buffer size:\t", lay_inf.loc[:, "img_buf_depth"].max(), "\n\n")

# Comparison LINA Python Reference vs Tensorflow Qkeras Reference

l_tmp = img_org

lina_ref_tmp = np.zeros(1)
tf_ref_result = list()
tf_ref = list()

fract_correction = 2 ** (FWL - 0)
# fract_correction = 1
# lina_ref_result = lina_float_result

# Inference of tensor flow layers and merging into lina reference
for j in range(number_layers):
    # lina_ref_tmp = lina_ref_result[j].reshape(lay_inf.loc[j, 'psum_size']) / fract_correction
    lina_float_tmp = lina_float_result[j].reshape(lay_inf.loc[j, "psum_size"]) / fract_correction
    # Extract Tensorflow reference
    if lay_inf.loc[j, "type"] == lay_name_conv:
        # Calc Layer Ref
        l_tmp = model_qkeras.layers[lay_inf.loc[j, "corresponding_layer"]](l_tmp)
        if lay_inf.loc[j, "activation"] == "relu":
            # assuming, that activation is directly the next layer
            l_tmp = model_qkeras.layers[lay_inf.loc[j, "corresponding_layer"] + 1](l_tmp)
        print(l_tmp.shape)
        print(lay_inf.loc[j, "psum_size"])
        ref_tmp = l_tmp.numpy().transpose(3, 1, 2, 0).reshape(lay_inf.loc[j, "psum_size"])

        tf_ref.append(np.c_[lina_float_tmp, ref_tmp, lina_float_tmp])
        tf_ref_result.append(l_tmp)
    if lay_inf.loc[j, "type"] == lay_name_dense:
        # because flatten layers are skipped in KUPEGA, they need to be considered when extracting from the orig model
        if lay_inf.loc[j, "config"] == "flatten":
            # Append flatten only optionally
            l_tmp = model_qkeras.layers[lay_inf.loc[j, "corresponding_layer"] - 1](l_tmp)
            # when there is only one layer, l_tmp needs still to be a tensorflow type instead of numpy
            if number_layers == 1 or j == 0:
                ref_tmp = l_tmp.reshape(lay_inf.loc[j, "img_size"])
            else:
                ref_tmp = l_tmp.numpy().reshape(lay_inf.loc[j, "img_size"])
            tf_ref.append(
                np.c_[
                    lina_float_result[j - 1].reshape(lay_inf.loc[j - 1, "psum_size"])
                    / fract_correction,
                    ref_tmp,
                    lina_float_result[j - 1].reshape(lay_inf.loc[j - 1, "psum_size"])
                    / fract_correction,
                ]
            )
            tf_ref_result.append(l_tmp)
            # Calc Layer Ref
            l_tmp = model_qkeras.layers[lay_inf.loc[j, "corresponding_layer"]](l_tmp)
            # tf_ind += 1
        else:
            l_tmp = model_qkeras.layers[lay_inf.loc[j, "corresponding_layer"]](l_tmp)
        if lay_inf.loc[j, "activation"] == "relu":
            # assuming, that acivation is directly the next layer
            l_tmp = model_qkeras.layers[lay_inf.loc[j, "corresponding_layer"] + 1](l_tmp)
        ref_tmp = l_tmp.numpy().transpose()
        tf_ref.append(np.c_[lina_float_tmp, ref_tmp, lina_float_tmp])
        tf_ref_result.append(l_tmp)

# Evaluation of lina vs tensorflow reference

print(f"Quantization Error:\t{float(2 ** (-FWL + 1)):{2}.{2}}")
plot_img = False
num_flatten = 0

for j in range(number_layers):
    print("Layer " + str(j) + ":\t\t" + lay_inf.loc[j, "type"])
    if (lay_inf.loc[j, "type"] == lay_name_conv) and plot_img:
        # lina data
        a = tf_ref[j][:, 0]  # .astype(np.float)
        # reference data
        b = tf_ref[j][:, 1]  # .astype(np.float)
        # Ob das bei assymmetrischen Kernels noch klappt, gilt es noch zu untersuchen
        img_a = a.reshape(lay_inf.loc[j, "M"], lay_inf.loc[j, "F"], lay_inf.loc[j, "E"])
        img_b = b.reshape(lay_inf.loc[j, "M"], lay_inf.loc[j, "F"], lay_inf.loc[j, "E"])

        # plotte den vergleich filterweise
        for i in range(lay_inf.loc[j, "M"]):
            """
            fig, (ax0, ax1) = plt.subplots(nrows=2)
            im0 = ax0.pcolormesh(img_a[i, :, :])
            fig.colorbar(im0, ax0)
            ax0.set_title("Memory")
            im1 = ax1.pcolormesh(img_b[i, :, :])
            fig.colorbar(im1, ax1)
            ax1.set_title("TF-Reference")
            """
            max_val = img_b[i, :, :].max()
            min_val = img_b[i, :, :].min()
            plt.subplot(1, 2, 1)
            plt.pcolormesh(img_a[i, :, :], vmin=min_val, vmax=max_val)
            plt.title("Memory")
            plt.subplot(1, 2, 2)
            plt.pcolormesh(img_b[i, :, :], vmin=min_val, vmax=max_val)
            plt.title("TF-Reference")

            plt.suptitle("Layer " + str(j))
            plt.savefig("plots/layer_" + str(j) + "_ch_" + str(i) + ".png")
            plt.close()
    # for flatten layers also the source data is compared against the reference
    if (lay_inf.loc[j, "type"] == lay_name_dense) and plot_img:
        if lay_inf.loc[j, "config"] == "flatten":
            # lina data
            a = tf_ref[j][:, 0].astype(float)
            # reference data
            b = tf_ref[j][:, 1].astype(float)
            # Ob das bei assymmetrischen Kernels noch klappt, gilt es noch zu untersuchen
            img_a = a.reshape(
                lay_inf.loc[j - 1, "M"],
                lay_inf.loc[j - 1, "F"],
                lay_inf.loc[j - 1, "E"],
            )
            img_b = b.reshape(
                lay_inf.loc[j - 1, "M"],
                lay_inf.loc[j - 1, "F"],
                lay_inf.loc[j - 1, "E"],
            )
            # img_a = a.reshape(lay_inf.loc[j - 1, 'M'], lay_inf.loc[j - 1, 'H'] - lay_inf.loc[j - 1, 'S'] + 1,
            #                   lay_inf.loc[j - 1, 'W'] - lay_inf.loc[j - 1, 'R'] + 1)
            # img_b = b.reshape(lay_inf.loc[j - 1, 'M'], lay_inf.loc[j - 1, 'H'] - lay_inf.loc[j - 1, 'S'] + 1,
            #                   lay_inf.loc[j - 1, 'W'] - lay_inf.loc[j - 1, 'R'] + 1)
            max_val = b.max()
            min_val = b.min()
            for i in range(lay_inf.loc[j - 1, "M"]):
                plt.subplot(1, 2, 1)
                plt.pcolormesh(img_a[i, :, :], vmin=min_val, vmax=max_val)
                plt.title("Memory")
                plt.subplot(1, 2, 2)
                plt.pcolormesh(img_b[i, :, :], vmin=min_val, vmax=max_val)
                plt.title("TF-Reference")

                plt.suptitle("Layer " + str(j))
                plt.savefig("plots/layer_flatten_" + str(j) + "_ch_" + str(i) + ".png")
                plt.close()
            num_flatten += 1
        # lina data
        a = tf_ref[j + num_flatten][:, 0].astype(float)
        # reference data
        b = tf_ref[j + num_flatten][:, 1].astype(float)

        plt.plot(a, label="LINA Py Ref")
        plt.plot(b, label="TF-Reference")
        plt.title("Dense Layer " + str(j + num_flatten))
        plt.legend()

        plt.savefig("plots/layer_" + str(j + num_flatten) + ".png", dpi=300)
        plt.close()

    # Fehlerrechnung
    # align data format
    elem_num = tf_ref[j].shape[0]
    tf_ref_data = np.zeros(elem_num)
    lina_fixed = np.zeros(elem_num)
    lina_float = np.zeros(elem_num)
    for i in range(elem_num):
        lina_float[i] = tf_ref[j][i, 2]
        tf_ref_data[i] = tf_ref[j][i, 1]
        lina_fixed[i] = tf_ref[j][i, 0]

    fixed_error = tf_ref_data - lina_fixed
    float_error = tf_ref_data - lina_float
    print("Error Tensorflow vs Lina Fixed Point")
    print(f"Max Error:\t\t{np.abs(fixed_error).max():{2}.{2}}")
    print(f"Mean Error:\t\t{np.abs(fixed_error).mean():{2}.{2}}")
    print(f"Median Error:\t{np.median(np.abs(fixed_error)):{2}.{2}}")
    print("Error Tensorflow vs Naive Quantized")
    print(f"Max Error:\t\t{np.abs(float_error).max():{2}.{2}}")
    print(f"Mean Error:\t\t{np.abs(float_error).mean():{2}.{2}}")
    print(f"Median Error:\t{np.median(np.abs(float_error)):{2}.{2}}\n")
