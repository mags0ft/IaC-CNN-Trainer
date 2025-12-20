"""
This script reads in all the config files from the folder cnn_images and writes the data to a new
file called uart_payload_config.txt. The same is done for the image and filter files. The script
then counts the amount of words in the file and writes the command byte, length and address to the
file. The command byte is the first byte of the payload, followed by the length of the payload and
the start address of the payload. It is then followed by the payload itself. The payload is written
continously to the memory of the FPGA using UART (in our case, the UI used is Cutecom).
"""

import os
import sys


def pad_to_four_bytes(my_string):
    """
    Pads the input string to 4 bytes length, inserting zeroes.
    """

    length = len(my_string)
    while length < 4:
        my_string = "0" + my_string
        length = len(my_string)
    return my_string


def convert_to_hex(my_string):
    """
    Converts a given binary (0 and 1) string to hexadecimal.
    """

    my_string = hex(int(my_string, 2))[2:]
    my_string = pad_to_four_bytes(my_string)
    my_string = bytes.fromhex(my_string)
    return my_string


def build_header(command_byte, length, address):
    """
    Constructs the header.
    """

    first_line = command_byte + length[:8]
    first_line = convert_to_hex(first_line)
    header = first_line

    second_line = length[8:16] + length[16:24]
    second_line = convert_to_hex(second_line)
    header += second_line

    third_line = address[:8] + address[8:16]
    third_line = convert_to_hex(third_line)
    header += third_line

    fourth_line = address[16:24] + address[24:32]
    fourth_line = convert_to_hex(fourth_line)
    header += fourth_line
    return header


# read in all files in the directory cnn_images that start with config and end with bin.txt
FOLDER_PATH = "data/scratchpad"

# first file name is config_l0_waveform.bin.txt the final name changes with the number of files
FILE_NAME = "config_l0_waveform_bin.txt"
AMOUNT_OF_LAYERS = int(sys.argv[1]) - 1


def main() -> None:
    """
    Main method of the script.
    """

    ensure_folders_exist()

    new_file = open("uart_payload/uart_payload_config.txt", "w")
    # this file will contain the data from all the config files

    # the loop shall run from 0 to the amount of layers (inclusive)
    for i in range(0, AMOUNT_OF_LAYERS + 1):
        file_name = "config_l" + str(i) + "_waveform_bin.txt"
        # open the file with file name from the folder cnn_images
        file_path = os.path.join(FOLDER_PATH, file_name)
        print("processing: " + file_path)
        with open(file_path, "r") as file:
            # read in all lines from the file and append them to the new file
            for line in file:
                new_file.write(line)
            file.close()

    new_file.close()

    # generate the command byte for uart
    # the command byte is the first byte of the payload it tells the UART what to do
    # THE MSB in the command byte is set to 0 to indicate a read, 1 indicates a write
    # the next byte is set to one to indicate a follow up command
    # the other 6 bits are reserved for future use
    # then follows the length of the payload, which are 3 bytes in total,
    # we send the bits 24 to 16 first, then 15 to 8, then 7 to 0
    # then follows the address of the register we want to write to which are 4 bytes in total
    # we send the bits 31 to 24 first, then 23 to 16, then 15 to 8, then 7 to 0
    # then follows the payload itself

    uart_data = open("uart_payload/uart_payload_config.txt", "r")
    uart_data_lines = uart_data.readlines()
    count = len(uart_data_lines)

    command_byte = "01000000"
    length = format(count, "024b")
    # config always starts at adress 0x00000000
    address = "00000000000000000000000000000000"
    uart_payload = open("uart_payload/uart_payload_config.hex", "wb")

    # always write 16 digits per line
    header = build_header(command_byte, length, address)
    uart_payload.write(header)

    for line in uart_data_lines:
        line = convert_to_hex(line)
        uart_payload.write(line)

    uart_data.close()
    uart_payload.close()

    # do the same for filter, but always remove the first line per file bevore concatenating
    filt_file_name = "filt_buf_l0_waveform_bin.txt"
    new_file = open("uart_payload/uart_payload_filt.txt", "w")
    for i in range(0, AMOUNT_OF_LAYERS + 1):
        filt_file_name = "filt_buf_l" + str(i) + "_waveform_bin.txt"
        file_path = os.path.join(FOLDER_PATH, filt_file_name)
        print("processing: " + file_path)
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines[1:]:
                new_file.write(line)
            file.close()
    new_file.close()
    # count the amount of words in the new_file
    uart_data = open("uart_payload/uart_payload_filt.txt", "r")
    uart_data_lines = uart_data.readlines()
    count = len(uart_data_lines)

    command_byte = "01000000"
    length = format(count, "024b")
    # filt always starts at adress 1024 + 131072 + 131072 which is in hex 0x00040400
    address = "00000000000001000000010000000000"
    uart_payload = open("uart_payload/uart_payload_filt.hex", "wb")

    # always write 16 digits per line
    header = build_header(command_byte, length, address)
    uart_payload.write(header)

    for line in uart_data_lines:
        line = convert_to_hex(line)
        uart_payload.write(line)

    uart_data.close()
    uart_payload.close()

    # for the image we only need the first one, the others are calculated by the fpga, and only here for reference
    img_file_name = "img_buf_l0_waveform_bin.txt"
    file_path = os.path.join(FOLDER_PATH, img_file_name)
    print("processing: " + file_path)
    uart_data = open(file_path, "r")
    uart_data_lines = uart_data.readlines()
    count = len(uart_data_lines) - 1

    # this is the last command byte, so no follow up command
    command_byte = "01000000"
    length = format(count, "024b")
    # image always starts at adress 1024 which is in hex 0x00000400
    address = "00000000000000000000010000000000"
    uart_payload = open("uart_payload/uart_payload_img.hex", "wb")

    # always write 16 digits per line
    header = build_header(command_byte, length, address)
    uart_payload.write(header)

    for line in uart_data_lines[1:]:
        line = convert_to_hex(line)
        uart_payload.write(line)

    uart_data.close()
    uart_payload.close()

    # generate a clean payload for each segment that overwrites everthing with the clean value
    cleanvalue = "dead\n"

    command_byte = "01000000"
    address = "00000000000000000000000000000000"
    length = format(1024, "024b")
    uart_payload = open("uart_payload/uart_payload_config_clean.hex", "wb")
    header = build_header(command_byte, length, address)
    uart_payload.write(header)
    for i in range(0, 1024):
        line = cleanvalue
        line = bytes.fromhex(line)
        uart_payload.write(line)
    uart_payload.close()

    command_byte = "01000000"
    address = "00000000000000000000010000000000"
    length = format(131072, "024b")
    uart_payload = open("uart_payload/uart_payload_img_clean.hex", "wb")
    header = build_header(command_byte, length, address)
    uart_payload.write(header)
    for i in range(0, 131072):
        line = cleanvalue
        line = bytes.fromhex(line)
        uart_payload.write(line)
    uart_payload.close()

    command_byte = "01000000"
    address = "00000000000000100000010000000000"
    length = format(131072, "024b")
    uart_payload = open("uart_payload/uart_payload_res_clean.hex", "wb")
    header = build_header(command_byte, length, address)
    uart_payload.write(header)
    for i in range(0, 131072):
        line = cleanvalue
        line = bytes.fromhex(line)
        uart_payload.write(line)
    uart_payload.close()

    command_byte = "01000000"
    address = "00000000000001000000010000000000"
    length = format(131072, "024b")
    uart_payload = open("uart_payload/uart_payload_filt_clean.hex", "wb")
    header = build_header(command_byte, length, address)
    uart_payload.write(header)
    for i in range(0, 131072):
        line = cleanvalue
        line = bytes.fromhex(line)
        uart_payload.write(line)
    uart_payload.close()


def ensure_folders_exist():
    """
    Ensures the `uart_payload` directory exists; creates it if not present.
    """

    if os.path.exists("uart_payload"):
        return

    print("creating folder uart_payload")
    os.makedirs("uart_payload")


if __name__ == "__main__":
    main()
