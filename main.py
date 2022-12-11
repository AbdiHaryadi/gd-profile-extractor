import cv2 as cv
from cv2 import Mat
import numpy as np
import csv

def extract_profile_data(fragment):
    fragments = extract_profile_data_image_fragments(fragment)

    result = {}
    for key, fragment in fragments.items():
        if key == "name":
            result[key] = predict_string_with_pusab(fragment)
        elif key == "global_rank":
            result[key] = predict_number_with_global_rank_font(fragment)
        else:
            result[key] = predict_number_with_pusab(fragment)

    if "cp" not in result.keys():
        result["cp"] = 0

    return result

def extract_profile_data_image_fragments(image):
    hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    gray_image = np.multiply(1 - hsv_image[:,:,1] / 255, hsv_image[:,:,2] / 255)
    gray_image = np.vectorize(lambda x: x ** (1/2))(gray_image)
    gray_image = (gray_image * 255).astype('uint8')
    

    _, thresholded_image = cv.threshold(gray_image, round(0.8 * 255), 255, cv.THRESH_BINARY)

    thresholded_image = extract_user_profile_fragment(thresholded_image)
    uniform_row_count = 128
    col_count = round(thresholded_image.shape[1] * (uniform_row_count / thresholded_image.shape[0]))
    thresholded_image = cv.resize(thresholded_image, (col_count, uniform_row_count))

    cv.imshow("Image", thresholded_image)

    row_midpoint = uniform_row_count // 2

    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (14, 2))
    kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (7, 2))
    top_dilate_result = cv.dilate(thresholded_image[:row_midpoint,:], kernel1)
    bottom_dilate_result = cv.dilate(thresholded_image[row_midpoint:,:], kernel2)

    dilate_result = thresholded_image.copy()
    dilate_result[:row_midpoint,:] = top_dilate_result
    dilate_result[row_midpoint:,:] = bottom_dilate_result

    # Analyze components
    analysis = cv.connectedComponentsWithStats(dilate_result, 4, cv.CV_32S)
    (label_count, label_ids, values, _) = analysis

    ## Clean small parts
    top_section_idx_list: list[int] = []
    bottom_section_idx_list: list[int] = []

    for idx in range(1, label_count):
        area = values[idx, cv.CC_STAT_AREA]
        left = values[idx, cv.CC_STAT_LEFT]
        top = values[idx, cv.CC_STAT_TOP]
        
        if left == 0:
            dilate_result[label_ids == idx] = 0
        elif top >= row_midpoint and area <= 200:
            dilate_result[label_ids == idx] = 0
        elif area <= 150:
            dilate_result[label_ids == idx] = 0
        else:
            if top < row_midpoint:
                top_section_idx_list.append(idx)
            else:
                bottom_section_idx_list.append(idx)

    ## Sort top section
    top_section_idx_list.sort(key=lambda x: values[x, cv.CC_STAT_LEFT])
    # dilate_result[label_ids == top_section_idx_list[0]] = 0 # "Global Rank" word

    # for i in range(3, len(top_section_idx_list)): # In the right of name
    #     dilate_result[label_ids == top_section_idx_list[i]] = 0

    # Clean bottom section
    # bottom_section_idx_list.sort(key=lambda x: centroid[x][0])
    bottom_section_idx_list.sort(key=lambda x: values[x, cv.CC_STAT_LEFT])
    second_element_left = values[bottom_section_idx_list[1], cv.CC_STAT_LEFT]
    first_element_idx = bottom_section_idx_list[0]
    first_element_right = values[first_element_idx, cv.CC_STAT_LEFT] + values[first_element_idx, cv.CC_STAT_WIDTH]
    ref_distance = second_element_left - first_element_right

    bottom_section_clean_completed = False
    i = 0

    distance_max_error = 0.06
    while i < len(bottom_section_idx_list) - 1 and not bottom_section_clean_completed:
        prev_idx = bottom_section_idx_list[i]
        prev_right = values[prev_idx, cv.CC_STAT_LEFT] + values[prev_idx, cv.CC_STAT_WIDTH]

        j = i + 1
        expected_distance_found = False
        while j < len(bottom_section_idx_list) and not expected_distance_found:
            curr_idx = bottom_section_idx_list[j]
            left = values[curr_idx, cv.CC_STAT_LEFT]
            distance = (left - prev_right) / ref_distance

            if distance >= 1 - distance_max_error:
                expected_distance_found = True
            else:
                # dilate_result[label_ids == curr_idx] = 0
                bottom_section_idx_list.pop(j)

        # j == len(bottom_section_idx) or distance >= 1

        if abs(distance - 1) > distance_max_error:
            # Clean all to the right
            # for idx in bottom_section_idx_list[j:]:
            #     dilate_result[label_ids == idx] = 0

            bottom_section_idx_list = bottom_section_idx_list[:j]
            bottom_section_clean_completed = True

        else:
            i += 1
    
    result = {
        "name": crop_by_component(thresholded_image, values, top_section_idx_list[2]),
        "global_rank": crop_by_component(thresholded_image, values, top_section_idx_list[1]),
        "stars": crop_by_component(thresholded_image, values, bottom_section_idx_list[0]),
        "diamonds": crop_by_component(thresholded_image, values, bottom_section_idx_list[1]),
        "secret_coins": crop_by_component(thresholded_image, values, bottom_section_idx_list[2]),
        "user_coins": crop_by_component(thresholded_image, values, bottom_section_idx_list[3]),
        "demons": crop_by_component(thresholded_image, values, bottom_section_idx_list[4])
    }

    if len(bottom_section_idx_list) == 6:
        result["cp"] = crop_by_component(thresholded_image, values, bottom_section_idx_list[5])

    return result

def extract_user_profile_fragment(thresholded_image):
    # Identify horizontal and vertical line for row and col min-max
    row_count, col_count = thresholded_image.shape
    quart_row_index = row_count // 4
    quart_col_index = col_count // 4
    row_min = np.argmax(np.sum(thresholded_image[:quart_row_index,:], axis=1))
    row_max = 3 * quart_row_index + np.argmax(np.sum(thresholded_image[3 * quart_row_index:,:], axis=1))
    col_min = np.argmax(np.sum(thresholded_image[:,:quart_col_index], axis=0))
    col_max = 3 * quart_col_index + np.argmax(np.sum(thresholded_image[:,3 * quart_col_index:], axis=0))

    # Divide the row by 4, take the first row
    row_max = row_min + (row_max - row_min) // 4

    return thresholded_image[row_min:row_max, col_min:col_max]


def crop_by_component(thresholded_image, values, idx):
    left = values[idx, cv.CC_STAT_LEFT]
    right = left + values[idx, cv.CC_STAT_WIDTH]
    top = values[idx, cv.CC_STAT_TOP]
    bottom = top + values[idx, cv.CC_STAT_HEIGHT]

    return thresholded_image[top:bottom, left:right]

def predict_string_with_pusab(image):
    analysis = cv.connectedComponentsWithStats(image, 4, cv.CV_32S)
    (label_count, _, values, _) = analysis

    idx_list = [i for i in range(1, label_count)]
    idx_list.sort(key=lambda x: values[x, cv.CC_STAT_LEFT])

    result = ""

    for idx in idx_list:
        left = values[idx, cv.CC_STAT_LEFT]
        right = left + values[idx, cv.CC_STAT_WIDTH]
        char_image = image[:,left:right]

        predict_char_result = predict_char_with_pusab(char_image)
        result += predict_char_result["value"]

    return result

def predict_number_with_pusab(image):
    analysis = cv.connectedComponentsWithStats(image, 4, cv.CV_32S)
    (label_count, _, values, _) = analysis

    idx_list = [i for i in range(1, label_count)]
    idx_list.sort(key=lambda x: values[x, cv.CC_STAT_LEFT])

    result = ""

    for idx in idx_list:
        left = values[idx, cv.CC_STAT_LEFT]
        right = left + values[idx, cv.CC_STAT_WIDTH]
        char_image = image[:,left:right]

        predict_char_result = predict_digit_with_pusab(char_image)
        result += predict_char_result["value"]

    return int(result)

def predict_number_with_pusab_with_second_method(image):
    uniform_row_count = 47
    template_map = {
        "0": cv.imread('dataset/fonts/pusabgd/0.png', cv.IMREAD_UNCHANGED),
        "1": cv.imread('dataset/fonts/pusabgd/1.png', cv.IMREAD_UNCHANGED),
        "2": cv.imread('dataset/fonts/pusabgd/2.png', cv.IMREAD_UNCHANGED),
        "3": cv.imread('dataset/fonts/pusabgd/3.png', cv.IMREAD_UNCHANGED),
        "4": cv.imread('dataset/fonts/pusabgd/4.png', cv.IMREAD_UNCHANGED),
        "5": cv.imread('dataset/fonts/pusabgd/5.png', cv.IMREAD_UNCHANGED),
        "6": cv.imread('dataset/fonts/pusabgd/6.png', cv.IMREAD_UNCHANGED),
        "7": cv.imread('dataset/fonts/pusabgd/7.png', cv.IMREAD_UNCHANGED),
        "8": cv.imread('dataset/fonts/pusabgd/8.png', cv.IMREAD_UNCHANGED),
        "9": cv.imread('dataset/fonts/pusabgd/9.png', cv.IMREAD_UNCHANGED),
    }
    segment_threshold = 0.91

    number_string, font_max_score = predict_global_rank_number_string_with_configuration(
        image,
        template_map=template_map,
        uniform_row_count=uniform_row_count,
        segment_threshold=segment_threshold
    )

    return int(number_string)

def predict_number_with_global_rank_font(image):
    config_list = [
        {
            "uniform_row_count": 60,
            "template_map": {
                "0": cv.imread('dataset/fonts/aller/0.png', cv.IMREAD_UNCHANGED),
                "1": cv.imread('dataset/fonts/aller/1.png', cv.IMREAD_UNCHANGED),
                "2": cv.imread('dataset/fonts/aller/2.png', cv.IMREAD_UNCHANGED),
                "3": cv.imread('dataset/fonts/aller/3.png', cv.IMREAD_UNCHANGED),
                "4": cv.imread('dataset/fonts/aller/4.png', cv.IMREAD_UNCHANGED),
                "5": cv.imread('dataset/fonts/aller/5.png', cv.IMREAD_UNCHANGED),
                "6": cv.imread('dataset/fonts/aller/6.png', cv.IMREAD_UNCHANGED),
                "7": cv.imread('dataset/fonts/aller/7.png', cv.IMREAD_UNCHANGED),
                "8": cv.imread('dataset/fonts/aller/8.png', cv.IMREAD_UNCHANGED),
                "9": cv.imread('dataset/fonts/aller/9.png', cv.IMREAD_UNCHANGED),
            },
            "segment_threshold": 0.75
        },

        {
            "uniform_row_count": 74,
            "template_map": {
                "0": cv.imread('dataset/fonts/arial/0.png', cv.IMREAD_UNCHANGED),
                "1": cv.imread('dataset/fonts/arial/1.png', cv.IMREAD_UNCHANGED),
                "2": cv.imread('dataset/fonts/arial/2.png', cv.IMREAD_UNCHANGED),
                "3": cv.imread('dataset/fonts/arial/3.png', cv.IMREAD_UNCHANGED),
                "4": cv.imread('dataset/fonts/arial/4.png', cv.IMREAD_UNCHANGED),
                "5": cv.imread('dataset/fonts/arial/5.png', cv.IMREAD_UNCHANGED),
                "6": cv.imread('dataset/fonts/arial/6.png', cv.IMREAD_UNCHANGED),
                "7": cv.imread('dataset/fonts/arial/7.png', cv.IMREAD_UNCHANGED),
                "8": cv.imread('dataset/fonts/arial/8.png', cv.IMREAD_UNCHANGED),
                "9": cv.imread('dataset/fonts/arial/9.png', cv.IMREAD_UNCHANGED),
            },
            "segment_threshold": 0.85
        },
    ]

    most_likely_number_string = ""
    template_max_score = 0.0

    for config in config_list:
        number_string, font_max_score = predict_global_rank_number_string_with_configuration(
            image,
            template_map=config["template_map"],
            uniform_row_count=config["uniform_row_count"],
            segment_threshold=config["segment_threshold"]
        )

        if font_max_score > template_max_score:
            template_max_score = font_max_score
            most_likely_number_string = number_string

    return int(most_likely_number_string)

def predict_global_rank_number_string_with_configuration(image, template_map, uniform_row_count, segment_threshold):
    resized_image = resize_by_row_count(image, row_count=uniform_row_count)

    union_match_result = np.zeros(resized_image.shape, dtype=np.double)
    col_max_match_result_map = {}
    for key, template in template_map.items():
        match_result = cv.matchTemplate(resized_image, template, cv.TM_CCORR_NORMED)

        row, col = template.shape
        row_min = row // 2
        row_max = row_min + match_result.shape[0]
        col_min = col // 2
        col_max = col_min + match_result.shape[1]

        union_match_result[row_min:row_max, col_min:col_max] = np.maximum(
            match_result,
            union_match_result[row_min:row_max, col_min:col_max]
        )

        col_max_match_result_map[key] = np.amax(match_result, axis=0)
        
    
    i = 0
    previously_low_score = True
    number_string = ""
    font_max_score = 0.0
    while i < union_match_result.shape[1]:
        score = np.max(union_match_result[:,i])
        currently_low_score = score < segment_threshold
        if currently_low_score != previously_low_score:
            local_max_score = score
            best_i = i

            stop = False
            while not stop:
                i += 1
                score = np.max(union_match_result[:,i])

                if score < segment_threshold:
                    stop = True
                elif score > local_max_score:
                    local_max_score = score
                    best_i = i
                # else: do nothing
            # score < segment_threshold

            # Now check what character represents in that position
            char_prediction = ""
            for key, match_result in col_max_match_result_map.items():
                col_offset = template_map[key].shape[1] // 2
                if best_i - col_offset < match_result.shape[0]:
                    if local_max_score == match_result[best_i - col_offset]:
                        char_prediction = key

            number_string += char_prediction
                
            if font_max_score < local_max_score:
                font_max_score = local_max_score

        i += 1
    return number_string,font_max_score

def resize_by_row_count(image, row_count):
    col_count = round(image.shape[1] * row_count / image.shape[0])
    return cv.resize(image, (col_count, row_count))

def predict_char_with_pusab(char_image: Mat):
    template_map = {
        "A": cv.imread('dataset/fonts/pusabgd/a_upper.png', cv.IMREAD_UNCHANGED),
        "B": cv.imread('dataset/fonts/pusabgd/b_upper.png', cv.IMREAD_UNCHANGED),
        "C": cv.imread('dataset/fonts/pusabgd/c_upper.png', cv.IMREAD_UNCHANGED),
        "D": cv.imread('dataset/fonts/pusabgd/d_upper.png', cv.IMREAD_UNCHANGED),
        "E": cv.imread('dataset/fonts/pusabgd/e_upper.png', cv.IMREAD_UNCHANGED),
        "F": cv.imread('dataset/fonts/pusabgd/f_upper.png', cv.IMREAD_UNCHANGED),
        "G": cv.imread('dataset/fonts/pusabgd/g_upper.png', cv.IMREAD_UNCHANGED),
        "H": cv.imread('dataset/fonts/pusabgd/h_upper.png', cv.IMREAD_UNCHANGED),
        "I": cv.imread('dataset/fonts/pusabgd/i_upper.png', cv.IMREAD_UNCHANGED),
        "J": cv.imread('dataset/fonts/pusabgd/j_upper.png', cv.IMREAD_UNCHANGED),
        "K": cv.imread('dataset/fonts/pusabgd/k_upper.png', cv.IMREAD_UNCHANGED),
        "L": cv.imread('dataset/fonts/pusabgd/l_upper.png', cv.IMREAD_UNCHANGED),
        "M": cv.imread('dataset/fonts/pusabgd/m_upper.png', cv.IMREAD_UNCHANGED),
        "N": cv.imread('dataset/fonts/pusabgd/n_upper.png', cv.IMREAD_UNCHANGED),
        "O": cv.imread('dataset/fonts/pusabgd/o_upper.png', cv.IMREAD_UNCHANGED),
        "P": cv.imread('dataset/fonts/pusabgd/p_upper.png', cv.IMREAD_UNCHANGED),
        "Q": cv.imread('dataset/fonts/pusabgd/q_upper.png', cv.IMREAD_UNCHANGED),
        "R": cv.imread('dataset/fonts/pusabgd/r_upper.png', cv.IMREAD_UNCHANGED),
        "S": cv.imread('dataset/fonts/pusabgd/s_upper.png', cv.IMREAD_UNCHANGED),
        "T": cv.imread('dataset/fonts/pusabgd/t_upper.png', cv.IMREAD_UNCHANGED),
        "U": cv.imread('dataset/fonts/pusabgd/u_upper.png', cv.IMREAD_UNCHANGED),
        "V": cv.imread('dataset/fonts/pusabgd/v_upper.png', cv.IMREAD_UNCHANGED),
        "W": cv.imread('dataset/fonts/pusabgd/w_upper.png', cv.IMREAD_UNCHANGED),
        "X": cv.imread('dataset/fonts/pusabgd/x_upper.png', cv.IMREAD_UNCHANGED),
        "Y": cv.imread('dataset/fonts/pusabgd/y_upper.png', cv.IMREAD_UNCHANGED),
        "Z": cv.imread('dataset/fonts/pusabgd/z_upper.png', cv.IMREAD_UNCHANGED),
        "a": cv.imread('dataset/fonts/pusabgd/a_lower.png', cv.IMREAD_UNCHANGED),
        "b": cv.imread('dataset/fonts/pusabgd/b_lower.png', cv.IMREAD_UNCHANGED),
        "c": cv.imread('dataset/fonts/pusabgd/c_lower.png', cv.IMREAD_UNCHANGED),
        "d": cv.imread('dataset/fonts/pusabgd/d_lower.png', cv.IMREAD_UNCHANGED),
        "e": cv.imread('dataset/fonts/pusabgd/e_lower.png', cv.IMREAD_UNCHANGED),
        "f": cv.imread('dataset/fonts/pusabgd/f_lower.png', cv.IMREAD_UNCHANGED),
        "g": cv.imread('dataset/fonts/pusabgd/g_lower.png', cv.IMREAD_UNCHANGED),
        "h": cv.imread('dataset/fonts/pusabgd/h_lower.png', cv.IMREAD_UNCHANGED),
        "i": cv.imread('dataset/fonts/pusabgd/i_lower.png', cv.IMREAD_UNCHANGED),
        "j": cv.imread('dataset/fonts/pusabgd/j_lower.png', cv.IMREAD_UNCHANGED),
        "k": cv.imread('dataset/fonts/pusabgd/k_lower.png', cv.IMREAD_UNCHANGED),
        "l": cv.imread('dataset/fonts/pusabgd/l_lower.png', cv.IMREAD_UNCHANGED),
        "m": cv.imread('dataset/fonts/pusabgd/m_lower.png', cv.IMREAD_UNCHANGED),
        "n": cv.imread('dataset/fonts/pusabgd/n_lower.png', cv.IMREAD_UNCHANGED),
        "o": cv.imread('dataset/fonts/pusabgd/o_lower.png', cv.IMREAD_UNCHANGED),
        "p": cv.imread('dataset/fonts/pusabgd/p_lower.png', cv.IMREAD_UNCHANGED),
        "q": cv.imread('dataset/fonts/pusabgd/q_lower.png', cv.IMREAD_UNCHANGED),
        "r": cv.imread('dataset/fonts/pusabgd/r_lower.png', cv.IMREAD_UNCHANGED),
        "s": cv.imread('dataset/fonts/pusabgd/s_lower.png', cv.IMREAD_UNCHANGED),
        "t": cv.imread('dataset/fonts/pusabgd/t_lower.png', cv.IMREAD_UNCHANGED),
        "u": cv.imread('dataset/fonts/pusabgd/u_lower.png', cv.IMREAD_UNCHANGED),
        "v": cv.imread('dataset/fonts/pusabgd/v_lower.png', cv.IMREAD_UNCHANGED),
        "w": cv.imread('dataset/fonts/pusabgd/w_lower.png', cv.IMREAD_UNCHANGED),
        "x": cv.imread('dataset/fonts/pusabgd/x_lower.png', cv.IMREAD_UNCHANGED),
        "y": cv.imread('dataset/fonts/pusabgd/y_lower.png', cv.IMREAD_UNCHANGED),
        "z": cv.imread('dataset/fonts/pusabgd/z_lower.png', cv.IMREAD_UNCHANGED),
        "0": cv.imread('dataset/fonts/pusabgd/0.png', cv.IMREAD_UNCHANGED),
        "1": cv.imread('dataset/fonts/pusabgd/1.png', cv.IMREAD_UNCHANGED),
        "2": cv.imread('dataset/fonts/pusabgd/2.png', cv.IMREAD_UNCHANGED),
        "3": cv.imread('dataset/fonts/pusabgd/3.png', cv.IMREAD_UNCHANGED),
        "4": cv.imread('dataset/fonts/pusabgd/4.png', cv.IMREAD_UNCHANGED),
        "5": cv.imread('dataset/fonts/pusabgd/5.png', cv.IMREAD_UNCHANGED),
        "6": cv.imread('dataset/fonts/pusabgd/6.png', cv.IMREAD_UNCHANGED),
        "7": cv.imread('dataset/fonts/pusabgd/7.png', cv.IMREAD_UNCHANGED),
        "8": cv.imread('dataset/fonts/pusabgd/8.png', cv.IMREAD_UNCHANGED),
        "9": cv.imread('dataset/fonts/pusabgd/9.png', cv.IMREAD_UNCHANGED),
    }

    return predict_char_with_predefined_pusab_template_map(char_image, template_map)

def predict_digit_with_pusab(char_image: Mat):
    template_map = {
        "0": cv.imread('dataset/fonts/pusabgd/0.png', cv.IMREAD_UNCHANGED),
        "1": cv.imread('dataset/fonts/pusabgd/1.png', cv.IMREAD_UNCHANGED),
        "2": cv.imread('dataset/fonts/pusabgd/2.png', cv.IMREAD_UNCHANGED),
        "3": cv.imread('dataset/fonts/pusabgd/3.png', cv.IMREAD_UNCHANGED),
        "4": cv.imread('dataset/fonts/pusabgd/4.png', cv.IMREAD_UNCHANGED),
        "5": cv.imread('dataset/fonts/pusabgd/5.png', cv.IMREAD_UNCHANGED),
        "6": cv.imread('dataset/fonts/pusabgd/6.png', cv.IMREAD_UNCHANGED),
        "7": cv.imread('dataset/fonts/pusabgd/7.png', cv.IMREAD_UNCHANGED),
        "8": cv.imread('dataset/fonts/pusabgd/8.png', cv.IMREAD_UNCHANGED),
        "9": cv.imread('dataset/fonts/pusabgd/9.png', cv.IMREAD_UNCHANGED),
    }

    return predict_char_with_predefined_pusab_template_map(char_image, template_map)

def predict_digit_with_aller(char_image: Mat):
    template_map = {
        "0": cv.imread('dataset/fonts/aller/0.png', cv.IMREAD_UNCHANGED),
        "1": cv.imread('dataset/fonts/aller/1.png', cv.IMREAD_UNCHANGED),
        "2": cv.imread('dataset/fonts/aller/2.png', cv.IMREAD_UNCHANGED),
        "3": cv.imread('dataset/fonts/aller/3.png', cv.IMREAD_UNCHANGED),
        "4": cv.imread('dataset/fonts/aller/4.png', cv.IMREAD_UNCHANGED),
        "5": cv.imread('dataset/fonts/aller/5.png', cv.IMREAD_UNCHANGED),
        "6": cv.imread('dataset/fonts/aller/6.png', cv.IMREAD_UNCHANGED),
        "7": cv.imread('dataset/fonts/aller/7.png', cv.IMREAD_UNCHANGED),
        "8": cv.imread('dataset/fonts/aller/8.png', cv.IMREAD_UNCHANGED),
        "9": cv.imread('dataset/fonts/aller/9.png', cv.IMREAD_UNCHANGED),
    }

    return predict_char_with_predefined_aller_template_map(char_image, template_map)

def predict_char_with_predefined_pusab_template_map(char_image, pusab_template_map):
    uniform_char_row_count = 47
    padded_col_count = 60

    return predict_char_with_predefined_template_map(char_image, pusab_template_map, uniform_char_row_count, padded_col_count)

def predict_char_with_predefined_aller_template_map(char_image, aller_template_map):
    uniform_char_row_count = 60
    padded_col_count = 120

    return predict_char_with_predefined_template_map(char_image, aller_template_map, uniform_char_row_count, padded_col_count)

def predict_char_with_predefined_template_map(char_image, template_map, uniform_char_row_count, padded_col_count):
    char_col_count = round(char_image.shape[1] * uniform_char_row_count / char_image.shape[0])
    resized_char_image = cv.resize(char_image, (char_col_count, uniform_char_row_count))

    padded_char_image = np.zeros((uniform_char_row_count, padded_col_count), dtype=np.uint8)
    padded_char_image[:uniform_char_row_count, :char_col_count] = resized_char_image

    # Read the char
    best_score = 0.0
    best_char = ""
    for key, template in template_map.items():
        padded_template = np.zeros((template.shape[0], padded_col_count), dtype=np.uint8)
        padded_template[:,:template.shape[1]] = template
        match_result = cv.matchTemplate(padded_char_image, padded_template, cv.TM_CCORR_NORMED)
        score = np.max(match_result)
        
        if score > best_score:
            best_score = score
            best_char = key

    return { "value": best_char, "score": best_score }

if __name__ == "__main__":
    ss_dir = "dataset/screenshots/"

    profile_map = {}
    with open("dataset/screenshots_label.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            profile = {}
            path = ss_dir + row["filename"]
            for key, value in row.items():
                if key != "filename":
                    if key == "name":
                        profile[key] = value
                    else:
                        profile[key] = int(value)

            profile_map[path] = profile

    for path, profile in profile_map.items():
        image = cv.imread(path, cv.IMREAD_UNCHANGED)
        prediction = extract_profile_data(image)

        print(path)
        print(prediction)
        print()

        for key, value in prediction.items():
            expected_data = profile[key]
            actual_data = prediction[key]
            
            if expected_data != actual_data:
                if type(expected_data) == str:
                    if expected_data.lower() != actual_data.lower():
                        print("[case insensitive] {} - {}: Expected {}, got {}".format(path, key, expected_data, actual_data))
                else:
                    print("{} - {}: Expected {}, got {}".format(path, key, expected_data, actual_data))
            
        k = cv.waitKey(0)

    cv.destroyAllWindows()
