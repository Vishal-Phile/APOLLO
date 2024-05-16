from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import matplotlib.pyplot as plt

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

# Function to process non-blue areas in the image
def process_non_blue(image_path, pixel_to_mm_conversion, print_output=False):
    original_image = cv2.imread(image_path)
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

    # Create masks for each color
    masks = [cv2.inRange(hsv_image, lower, upper) for lower, upper in zip(lower_ranges, upper_ranges)]

    # Combine masks
    unmasked_regions = masks[0] | masks[1] | masks[2] | masks[3]

    all_contours, _ = cv2.findContours(unmasked_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [contour for contour in all_contours if big_enough(contour)]
    hull = cv2.convexHull(np.vstack(contours))

    # Extract the region within the contours
    x, y, w, h = cv2.boundingRect(hull)
    region_within_contours = original_image[y:y+h, x:x+w]

    return region_within_contours

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
def plot_combined_variation(data, title, palette):
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

    plt.figure(figsize=(10, 8))
    for i, color in enumerate(palette):
        counts = [0] * len(combined_data)
        for j, row in enumerate(combined_data):
            for col_color, col_count in row:
                if col_color == i:
                    counts[j] += col_count
        rgba_color = color / 255.0
        plt.bar(range(len(combined_data)), counts, color=rgba_color, label=f'Color {i}')

    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.show()


# Define your color palette
palette = [
    np.array([0xff, 0x00, 0x00]),
    np.array([0xff, 0x5d, 0x00]),
    np.array([0xff, 0xb9, 0x00]),
    np.array([0xe8, 0xff, 0x00]),
    np.array([0x8b, 0xff, 0x00]),
    np.array([0x2e, 0xff, 0x00]),
    np.array([0x00, 0xff, 0x2e]),
    np.array([0x00, 0xff, 0x8b]),
    np.array([0x00, 0xff, 0xe8]),
    np.array([0x00, 0xb9, 0xff]),
    np.array([0x00, 0x5d, 0xff]),
    np.array([0x00, 0x00, 0xff]),
]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process/")
async def process(file: UploadFile = File(...), pixel_to_mm_conversion: float = Form(...)):
    # Save the uploaded file
    file_path = f'static/images/{file.filename}'
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Load your image and process only the region within the contours
    region_within_contours = process_non_blue(file_path, pixel_to_mm_conversion)

    # Assign each pixel to the nearest color in the palette
    row_colors, col_colors = create_color_lists(region_within_contours, palette)

    # Plot combined variation row-wise and column-wise
    plot_combined_variation(row_colors, 'Combined Row-wise Color Variation within Contours', palette)
    plot_combined_variation(col_colors, 'Combined Column-wise Color Variation within Contours', palette)

    return {"file_path": file_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
