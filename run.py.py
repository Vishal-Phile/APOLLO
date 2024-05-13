import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE_CONSTANT = 20
ASPECT_RATIO_CONSTANT = 0.1

def aspect_ratio(contour):
    *_, width, height = cv2.boundingRect(contour)
    return width / height

def big_enough(contour):
    if contour.size >= SIZE_CONSTANT and aspect_ratio(contour) >= ASPECT_RATIO_CONSTANT:
        return True
    return False

def count_pixels(mask):
    return np.sum(mask > 0)

def calculate_pressure_areas(hsv_image, lower_ranges, upper_ranges, pixel_to_mm_conversion):
    pressure_areas = []
    color_areas = []
    pixel_counts = []
    for lower, upper in zip(lower_ranges, upper_ranges):
        mask = cv2.inRange(hsv_image, lower, upper)
        pixel_count = count_pixels(mask)
        pixel_counts.append(pixel_count)
        pressure_area = pixel_count * pixel_to_mm_conversion ** 2
        pressure_areas.append(pressure_area)
        color_area = np.sum(mask) * pixel_to_mm_conversion ** 2
        color_areas.append(color_area)
    return pressure_areas, color_areas, pixel_counts

def process_non_blue(image, pixel_to_mm_conversion, print_output=False):
    
    original_image = image
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Define the color ranges for red, orange, yellow, and green
    lower_ranges = [
        np.array([0, 50, 50]),   # Red
        np.array([11, 50, 50]),  # Orange
        np.array([20, 100, 100]), # Yellow
        np.array([35, 50, 50])   # Green
    ]
    
    upper_ranges = [
        np.array([10, 255, 255]), # Red
        np.array([20, 255, 255]), # Orange
        np.array([60, 255, 255]), # Yellow
        np.array([90, 255, 255])  # Green
    ]
    
    # Calculate pressure areas, color areas, and pixel counts for each color range
    pressure_areas, color_areas, pixel_counts = calculate_pressure_areas(hsv_image, lower_ranges, upper_ranges, pixel_to_mm_conversion)
    
    # Print the results on the console
    if print_output:
        st.write("Pressure Areas for Each Color Range:")
        for i, area in enumerate(pressure_areas):
            st.write(f"  {['Red', 'Orange', 'Yellow', 'Green'][i]}: {area:.3f} square mm")
        
        st.write("\nColor Areas for Each Color Range:")
        for i, area in enumerate(color_areas):
            st.write(f"  {['Red', 'Orange', 'Yellow', 'Green'][i]}: {area:.3f} square mm")
        
        st.write("\nPixel Counts for Each Color Range:")
        for i, count in enumerate(pixel_counts):
            st.write(f"  {['Red', 'Orange', 'Yellow', 'Green'][i]}: {count}")
    
    # Create masks for each color
    masks = [cv2.inRange(hsv_image, lower, upper) for lower, upper in zip(lower_ranges, upper_ranges)]

    # Combine masks
    unmasked_regions = masks[0] | masks[1] | masks[2] | masks[3]
    result_image = cv2.bitwise_and(original_image, original_image, mask=unmasked_regions)

    all_contours, _ = cv2.findContours(unmasked_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [contour for contour in all_contours if big_enough(contour)]
    hull = cv2.convexHull(np.vstack(contours))
    
    # Plotting the stacked bar graph
    colors = ['red', 'orange', 'yellow', 'green']
    plt.bar(colors, pressure_areas, color=colors)
    plt.xlabel('Color')
    plt.ylabel('Pressure Area (square mm)')
    plt.title('Pressure Areas for Each Color Range')
    plt.legend(['Red', 'Orange', 'Yellow', 'Green'])
    st.pyplot(plt)

    contour_image = np.zeros_like(original_image)  # Blank canvas for drawing contours

    cv2.drawContours(contour_image, [hull], 0, (0, 255, 0), 2)

    x, y, w, h = cv2.boundingRect(hull)
    cv2.line(contour_image, (x, y + h // 2), (x + w, y + h // 2), (0, 0, 255), 2)
    cv2.line(contour_image, (x + w // 2, y), (x + w // 2, y + h), (0, 0, 255), 2)

    if print_output:
        length_mm = w * pixel_to_mm_conversion
        width_mm = h * pixel_to_mm_conversion
        area_entire_polygon_mm2 = cv2.contourArea(hull) * pixel_to_mm_conversion ** 2
        area_sum_sub_strips_mm2 = sum(
            [cv2.contourArea(cv2.convexHull(contour)) * pixel_to_mm_conversion ** 2 for contour in contours])

        st.write(f"\nEntire Footprint:")
        st.write(f"  Length     : {length_mm:8.3f} mm")
        st.write(f"  Width      : {width_mm:8.3f} mm")
        st.write(f"  Net Area   : {area_sum_sub_strips_mm2:8.3f} square mm")
        st.write(f"  Gross Area : {area_entire_polygon_mm2:8.3f} square mm")
        st.write(f"  Land Ratio : {area_sum_sub_strips_mm2 / area_entire_polygon_mm2:8.2%}")

    for i, contour in enumerate(contours, start=1):
        hull_sub_strip = cv2.convexHull(contour)
        cv2.drawContours(contour_image, [hull_sub_strip], 0, (255, 255, 255), 2)

        if print_output:
            x, y, w, h = cv2.boundingRect(hull_sub_strip)
            length_mm = h * pixel_to_mm_conversion
            width_mm = w * pixel_to_mm_conversion
            area_sub_strip_mm2 = cv2.contourArea(hull_sub_strip) * pixel_to_mm_conversion ** 2

            st.write(f"\nSub-Strip {i} | Size = {contour.size:3d} | Aspect Ratio = {aspect_ratio(contour):.3f}")
            st.write(f"  Length : {length_mm:8.3f} mm")
            st.write(f"  Width  : {width_mm:8.3f} mm")
            st.write(f"  Area   : {area_sub_strip_mm2:8.3f} square mm")

    st.image(original_image, caption='Original Image', use_column_width=True)
    st.image(result_image, caption='Non-Blue Portions with Polygons', use_column_width=True)
    st.image(contour_image, caption='Contours', use_column_width=True)

def distance_from_origin(x, y):
    return np.sqrt(x**2 + y**2)

def assign_pixel_to_nearest_color(pixel, palette):
    distances = np.linalg.norm(np.array(palette) - pixel, axis=1)
    nearest_index = np.argmin(distances)
    return nearest_index

def create_color_lists(image, palette):
    height, width, _ = image.shape
    row_colors = [[] for _ in range(height)]
    col_colors = [[] for _ in range(width)]

    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            nearest_color_index = assign_pixel_to_nearest_color(pixel, palette)
            row_colors[y].append(nearest_color_index)
            col_colors[x].append(nearest_color_index)

    return row_colors, col_colors


def plot_stacked_bar(data, title, x_labels):
    colors = ['#FF0000', '#FF5D00', '#FFB900', '#E8FF00', '#8BFF00', '#2EFF00', '#00FF2E', '#00FF8B', '#00FFE8', '#00B9FF', '#005DFF', '#0000FF']
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(data[0]))
    try:
        for i, row in enumerate(data):
            plt.bar(x_labels, row, bottom=bottom, color=colors[i], label=f"Color {i}")
            bottom += row
        plt.xlabel('Distance from Origin')
        plt.ylabel('Count')
        plt.title(title)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
    except Exception as e:
        st.write("Error occurred while plotting stacked bar graph:", e)


def main():
    st.title("Footprint Analysis")
    uploaded_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

    if uploaded_image is not None:
        st.write("Uploaded image:")
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        pixel_to_mm_conversion = st.slider("Pixel to mm Conversion", min_value=0.01, max_value=1.0, step=0.01, value=0.1)

        process_non_blue(image, pixel_to_mm_conversion, print_output=True)

        palette = [
            np.array([0xff, 0x00, 0x00]),  # Red
            np.array([0xff, 0x5d, 0x00]),  # Orange
            np.array([0xff, 0xb9, 0x00]),  # Yellow
            np.array([0xe8, 0xff, 0x00]),  # Yellow-Green
            np.array([0x8b, 0xff, 0x00]),  # Lime Green
            np.array([0x2e, 0xff, 0x00]),  # Green
            np.array([0x00, 0xff, 0x2e]),  # Green-Cyan
            np.array([0x00, 0xff, 0x8b]),  # Cyan
            np.array([0x00, 0xff, 0xe8]),  # Cyan-Blue
            np.array([0x00, 0xb9, 0xff]),  # Sky Blue
            np.array([0x00, 0x5d, 0xff]),  # Deep Sky Blue
            np.array([0x00, 0x00, 0xff])   # Blue
        ]

        row_colors, col_colors = create_color_lists(image, palette)

        st.write("Row-wise Color Variation:")
        plot_stacked_bar(row_colors, 'Row-wise Color Variation', list(range(len(row_colors[0]))))

        st.write("Column-wise Color Variation:")
        plot_stacked_bar(col_colors, 'Column-wise Color Variation', list(range(len(col_colors[0]))))

if __name__ == "__main__":
    main()
