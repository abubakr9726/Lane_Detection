import yaml


def config():
    with open("config.yml", "r") as f:
        try:
            data = str(yaml.safe_load(f)).split("'")
            if data[37] == "True":
                GPU_enable = True
            else:
                GPU_enable = False
            if data[41] == "True":
                write_conf = True
            else:
                write_conf = False
            if data[45] == "True":
                Lane_detection = True
            else:
                Lane_detection = False
            if data[49] == "True":
                Object_detection = True
            else:
                Object_detection = False

            data_array = [data[5], data[9], data[13], data[17], data[21], float(data[25]), float(data[29]),
                          data[33], GPU_enable, write_conf, Lane_detection, Object_detection]
            print(data_array)
            # INPUT_FILE = data[5]
            # LABELS_FILE = data[9]
            # CONFIG_FILE = data[13]
            # WEIGHTS_FILE = data[17]
            # lane_detection_model = data[21]
            # CONFIDENCE_THRESHOLD = data[25]
            # nms_thresh = data[29]
            # vid_output = data[33]
            # GPU_enable= data[37]
            # write_conf= data[41]
            # Lane_detection = data[45]
            # Object_detection = data[49]
            # for i in range(len(data_array)):
            #      print(f"{i}--> {type(data_array[i])} --> {data_array[i]}")
        except yaml.YAMLError as exc:
            print(exc)
    return data_array

#config()

