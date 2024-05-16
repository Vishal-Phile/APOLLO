import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE_CONSTANT = 20
ASPECT_RATIO_CONSTANT = 0.1

# Function to calculate aspect ratio of a contour
def aspect_ratio(contour):
    *_, width, height = cv2.boundingRect(contour)
    return width / height

# Function to check if contour is big enough
def big_enough(contour):
    if contour.size >= SIZE_CONSTANT and aspect_ratio(contour) >= ASPECT_RATIO_CONSTANT:
        return True
    return False

# Function to count pixels in a mask
def count_pixels(mask):
    return np.sum(mask > 0)

# Function to calculate pressure areas and color areas
def calculate_areas(hsv_image, lower_ranges, upper_ranges, pixel_to_mm_conversion):
    areas = []
    pixel_counts = []
    for lower, upper in zip(lower_ranges, upper_ranges):
        mask = cv2.inRange(hsv_image, lower, upper)
        pixel_count = count_pixels(mask)
        pixel_counts.append(pixel_count)
        area = pixel_count * pixel_to_mm_conversion ** 2
        areas.append(area)
    return areas, pixel_counts

# Function to process non-blue areas in the image
def process_non_blue(image, pixel_to_mm_conversion):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
    areas, pixel_counts = calculate_areas(hsv_image, lower_ranges, upper_ranges, pixel_to_mm_conversion)
    
    # Create masks for each color
    masks = [cv2.inRange(hsv_image, lower, upper) for lower, upper in zip(lower_ranges, upper_ranges)]

    # Combine masks
    unmasked_regions = masks[0] | masks[1] | masks[2] | masks[3]

    all_contours, _ = cv2.findContours(unmasked_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [contour for contour in all_contours if big_enough(contour)]
    hull = cv2.convexHull(np.vstack(contours))

    # Draw contours on the original image
    cv2.drawContours(image, [hull], -1, (0, 255, 0), 2)

    # Extract the region within the contours
    x, y, w, h = cv2.boundingRect(hull)
    region_within_contours = image[y:y+h, x:x+w]

    return region_within_contours, hull, pixel_counts

# Function to assign each pixel to the nearest color from the palette
def assign_pixel_to_nearest_color(pixel, palette):
    distances = np.linalg.norm(np.array(palette) - pixel, axis=1)
    nearest_index = np.argmin(distances)
    return nearest_index

# Function to create lists of colors along rows and columns
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

# Function to plot variation graph
def plot_combined_variation(data, title, palette, color_names):
    combined_data = []
    current_color = None
    for row in data:
        combined_row = []
        count = 0
        for color in row:
            if color == current_color:
                count += 1
            else:
                if current_color is not None:
                    combined_row.append((current_color, count))
                current_color = color
                count = 1
        combined_row.append((current_color, count))
        combined_data.append(combined_row)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, color in enumerate(palette):
        counts = [0] * len(combined_data)
        for j, row in enumerate(combined_data):
            for col_color, col_count in row:
                if col_color == i:
                    counts[j] += col_count
        rgba_color = color / 255.0
        ax.bar(range(len(combined_data)), counts, color=rgba_color, label=color_names[i])

    ax.set_xlabel('Position')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    return fig

def main():
    st.set_page_config(layout="wide")
    st.title('Tire Footprint Color Variation Analysis')

    # Sidebar
    st.sidebar.title('Options')
    tabs = ["Contour and Areas", "Graph"]
    choice = st.sidebar.selectbox("Choose Tab", tabs)
    
    uploaded_file = st.sidebar.file_uploader("Upload a tire footprint image", type=["png", "jpg", "jpeg"])
    pixel_to_mm_conversion = st.sidebar.slider("Pixel to mm Conversion", min_value=0.1, max_value=1.0, step=0.1, value=0.1)

    # Main working area
    with st.sidebar:
        st.write("---")
        st.write("This is the working area")
        st.write("---")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        if choice == "Contour and Areas":
            if st.sidebar.button('Analyze'):
                # Process the uploaded image
                processed_image, hull, pixel_counts = process_non_blue(image, pixel_to_mm_conversion)

                st.subheader("Contour Image")
                st.image(processed_image, caption='Contour Image', use_column_width=True)

                st.subheader("Contour Area")
                contour_area = cv2.contourArea(hull)
                st.write(f"The contour area is {contour_area} square pixels.")

                st.subheader("Pixel Counts for Each Color")
                st.write("Red: ", pixel_counts[0])
                st.write("Orange: ", pixel_counts[1])
                st.write("Yellow: ", pixel_counts[2])
                st.write("Green: ", pixel_counts[3])

        elif choice == "Graph":
            if st.sidebar.button('Analyze'):
                # Process the uploaded image
                processed_image = process_non_blue(image, pixel_to_mm_conversion)[0]

                # Define color palette
                palette = [
                    np.array([0xff, 0x00, 0x00]), # Red
                    np.array([0xff, 0x5d, 0x00]), # Orange
                    np.array([0xff, 0xb9, 0x00]), # Yellow
                    np.array([0xe8, 0xff, 0x00]), # Yellow-Green
                    np.array([0x8b, 0xff, 0x00]), # Green
                    np.array([0x2e, 0xff, 0x00]), # Chartreuse
                    np.array([0x00, 0xff, 0x2e]), # Green
                    np.array([0x00, 0xff, 0x8b]), # Turquoise
                    np.array([0x00, 0xff, 0xe8]), # Aquamarine
                    np.array([0x00, 0xb9, 0xff]), # Sky Blue
                    np.array([0x00, 0x5d, 0xff]), # Navy
                    np.array([0x00, 0x00, 0xff])  # Indigo
                ]
                
                # Define color names
                color_names = ["Crimson", "Tangerine", "Mustard", "Lemon", "Lime", "Chartreuse", 
                               "Green", "Turquoise", "Aquamarine", "Sky Blue", "Navy", "Indigo"]

                # Assign each pixel to the nearest color in the palette
                row_colors, col_colors = create_color_lists(processed_image, palette)

                # Plot combined variation row-wise and column-wise
                st.subheader('Combined Row-wise Color Variation within Contours')
                row_fig = plot_combined_variation(row_colors, 'Combined Row-wise Color Variation within Contours', palette, color_names)
                st.pyplot(row_fig)

                st.subheader('Combined Column-wise Color Variation within Contours')
                col_fig = plot_combined_variation(col_colors, 'Combined Column-wise Color Variation within Contours', palette, color_names)
                st.pyplot(col_fig)

if __name__ == "__main__":
    main()
